"""
TranSplat ↔ 4DGS harmonization bridge.

This module connects TranSplat-style SH lighting transfer (Phase 2) with the
image-driven harmonizer + ΔSH optimization in ``optimize_sh.py``.

**Mask quality (critical):** TranSplat math assumes object Gaussians are
correctly segmented. Poor ``mask_table`` / ``object_mask`` causes wrong
splits, wrong Phase-2 reference SH, and can *amplify* halos. Fix masks before
expecting physics-style priors to help.

**Phase 1 (V_lm) visibility baking:** TranSplat's ``phase_1_bake_visibility_sh``
expects the rasterizer to return ``depth`` or ``surf_depth`` (see
``check_transsplat_phase1_depth_support()``). The default sa4d
``gaussian_renderer.render`` does *not* expose those keys; use a compatible
rasterizer fork, or supply a pre-baked ``V_lm`` tensor (``.pt``) from an
external TranSplat Phase-1 run.

Scene lighting in SH (``L_lm``) is obtained from HDR env maps via the same
projection as ``radiance_transfer_TranSplat.py`` (Monte Carlo directions +
``compute_global_sh_coeffs``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


def _load_transsplat_radiance_module():
    """Load ``transsplat/radiance_transfer_TranSplat.py`` as a standalone module."""
    path = os.path.join(SA4D_ROOT, "transsplat", "radiance_transfer_TranSplat.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TranSplat radiance module not found: {path}")
    spec = importlib.util.spec_from_file_location("transsplat_radiance_transfer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def check_transsplat_phase1_depth_support() -> Dict[str, Any]:
    """
    Verify whether ``gaussian_renderer.render`` exposes ``depth`` / ``surf_depth``
    keys required by TranSplat ``phase_1_bake_visibility_sh``.

    The stock sa4d ``render()`` return dict only includes ``render``, ``mask``,
    ``viewspace_points``, ``visibility_filter``, ``radii``, ``deformed_points``,
    ``points2d`` — no per-pixel depth map for visibility baking.

    Returns:
        dict with keys: ``supported`` (bool), ``missing_keys`` (list), ``message`` (str).
    """
    gr_path = os.path.join(SA4D_ROOT, "gaussian_renderer", "__init__.py")
    render_return_keys = [
        "render",
        "mask",
        "viewspace_points",
        "visibility_filter",
        "radii",
        "deformed_points",
        "points2d",
    ]
    try:
        with open(gr_path, "r", encoding="utf-8") as f:
            body = f.read()
    except OSError as e:
        return {
            "supported": False,
            "render_keys": render_return_keys,
            "missing_keys": ["depth", "surf_depth"],
            "message": f"Could not read {gr_path}: {e}",
        }

    # Main ``render()`` return block: first occurrence of return {"render": rendered_image,
    supported = False
    m = re.search(
        r"return\s*\{[^\}]*?\"render\"\s*:\s*rendered_image[^\}]*\}",
        body,
        flags=re.DOTALL,
    )
    if m:
        blk = m.group(0)
        supported = ("\"depth\"" in blk or "'depth'" in blk or "depth:" in blk or "surf_depth" in blk)

    msg = (
        "Phase 1 visibility baking appears supported (depth/surf_depth in main render return dict)."
        if supported
        else (
            "Phase 1 visibility baking is NOT supported in this build: ``render()`` does not "
            "expose ``depth`` / ``surf_depth`` (see ``transsplat/radiance_transfer_TranSplat.py`` "
            "``phase_1_bake_visibility_sh``). Use ``--transsplat_vlm_pt`` with precomputed V_lm, "
            "or ``--transsplat_skip_visibility`` for an unobstructed proxy (approximate)."
        )
    )
    return {
        "supported": bool(supported),
        "render_keys": render_return_keys,
        "missing_keys": [] if supported else ["depth", "surf_depth"],
        "message": msg,
    }


def load_hdr_env_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load an HDR/LDR environment map as float RGB on ``device`` [H,W,3]."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV (cv2) is required to load HDR env maps for TranSplat bridge") from e
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread failed for {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32)).to(device)


def compute_env_sh_pair(
    hdr_source_path: str,
    hdr_target_path: str,
    num_samples: int,
    l_max: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """
    Project source/target env maps to SH coefficients ``L_lm_*`` [K,3].

    Returns:
        L_lm_source, L_lm_target, sh_coeffs_offset (gray 0.5 map), and a bundle
        ``ts_bundle`` of precomputed tensors for Phase 2:
        directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A,
        gaunt_tensor, AT_A_inv
    """
    ts = _load_transsplat_radiance_module()
    directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A, gaunt_tensor = ts.precompute_sh_sampling(
        int(num_samples), int(l_max), device
    )
    AT_A_inv = torch.linalg.inv(AT_A)

    src = load_hdr_env_tensor(hdr_source_path, device)
    tgt = load_hdr_env_tensor(hdr_target_path, device)
    if tgt.shape[:2] != src.shape[:2]:
        import cv2

        tgt = torch.from_numpy(
            cv2.resize(
                tgt.detach().cpu().numpy(),
                (src.shape[1], src.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
        ).to(device)

    offset_gpu = torch.full_like(tgt, 0.5)
    L_lm_source = ts.compute_global_sh_coeffs(src, directions, sqrt_weights, A_weighted, AT_A_weighted)
    L_lm_target = ts.compute_global_sh_coeffs(tgt, directions, sqrt_weights, A_weighted, AT_A_weighted)
    sh_coeffs_offset = ts.compute_global_sh_coeffs(offset_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)

    bundle = {
        "directions": directions,
        "sqrt_weights": sqrt_weights,
        "A": A,
        "A_weighted": A_weighted,
        "AT_A_weighted": AT_A_weighted,
        "AT_A": AT_A,
        "gaunt_tensor": gaunt_tensor,
        "AT_A_inv": AT_A_inv,
        "l_max": int(l_max),
        "K": int(A_weighted.shape[1]),
    }
    return L_lm_source, L_lm_target, sh_coeffs_offset, bundle


def unobstructed_v_lm(N: int, K: int, device: torch.device) -> torch.Tensor:
    """
    Crude visibility SH proxy: energy only in the DC visibility band.

    This is **not** physically accurate self-shadowing; it allows Phase 2 to
    run when real ``V_lm`` is unavailable (see ``check_transsplat_phase1_depth_support``).
    """
    V = torch.zeros(N, K, device=device, dtype=torch.float32)
    V[:, 0] = 1.0
    return V


def gaussians_object_to_transsplat_dict(gaussians, object_mask: torch.Tensor) -> Dict[str, Any]:
    """Build a TranSplat-style ``object_gaussians`` dict from a slice of ``GaussianModel``."""
    ts = _load_transsplat_radiance_module()
    dc = gaussians._features_dc[object_mask].detach().clone()
    rest = gaussians._features_rest[object_mask].detach().clone()
    xyz = gaussians._xyz[object_mask].detach().clone()
    op = gaussians._opacity[object_mask].detach().clone()
    sc = gaussians._scaling[object_mask].detach().clone()
    rot = gaussians._rotation[object_mask].detach().clone()
    l_max = int(gaussians.max_sh_degree)
    Rm = ts.quaternion2rotmat(F.normalize(rot, dim=1))
    normals = F.normalize(Rm[..., 2], dim=1)
    return {
        "xyz": torch.nn.Parameter(xyz),
        "features_dc": torch.nn.Parameter(dc),
        "features_rest": torch.nn.Parameter(rest),
        "opacity": torch.nn.Parameter(op),
        "scaling": torch.nn.Parameter(sc),
        "rotation": torch.nn.Parameter(rot),
        "normal": torch.nn.Parameter(normals),
        "max_sh_degree": l_max,
    }


def compute_transsplat_phase2_reference_sh(
    gaussians,
    object_mask: torch.Tensor,
    L_lm_source: torch.Tensor,
    L_lm_target: torch.Tensor,
    sh_coeffs_offset: torch.Tensor,
    ts_bundle: Dict[str, Any],
    V_lm: torch.Tensor,
    floor_alpha: float = 0.05,
    tau_max: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run TranSplat Phase 2 on a **copy** of object Gaussians; return reference
    ``features_dc`` and ``features_rest`` [N_obj, ...] (detached).
    """
    ts = _load_transsplat_radiance_module()
    obj = gaussians_object_to_transsplat_dict(gaussians, object_mask)
    all_normals = F.normalize(obj["normal"].detach(), dim=1)
    l_max = int(obj["max_sh_degree"])
    device = obj["xyz"].device
    directions = ts_bundle["directions"]
    A = ts_bundle["A"]
    AT_A_inv = ts_bundle["AT_A_inv"]
    gaunt_tensor = ts_bundle["gaunt_tensor"]

    with torch.no_grad():
        ts.phase_2_decoupled_relight(
            obj,
            all_normals,
            V_lm,
            L_lm_source,
            L_lm_target,
            sh_coeffs_offset,
            A,
            AT_A_inv,
            gaunt_tensor,
            directions,
            l_max,
            device,
            floor_alpha=float(floor_alpha),
            tau_max=float(tau_max),
        )
    ref_dc = obj["features_dc"].detach().clone()
    ref_rest = obj["features_rest"].detach().clone()
    return ref_dc, ref_rest


def load_v_lm(path: str, N_expected: int, K_expected: int, device: torch.device) -> torch.Tensor:
    V = torch.load(path, map_location=device)
    if not torch.is_tensor(V):
        V = torch.tensor(V, device=device, dtype=torch.float32)
    V = V.to(device=device, dtype=torch.float32)
    if V.shape[0] != N_expected or V.shape[1] != K_expected:
        raise ValueError(f"V_lm shape {tuple(V.shape)} != expected ({N_expected}, {K_expected})")
    return V


def band_weighted_delta_rest_regularizer(delta_rest: torch.Tensor, l_max: int, band_weights: Optional[list] = None) -> torch.Tensor:
    """
    TranSplat-inspired band structure: higher SH bands get stronger L2 penalty.

    ``delta_rest`` shape [N, (l_max+1)^2 - 1, 3] in the usual 3DGS layout.
    """
    if band_weights is None:
        # Default: ramp weights for l=1..l_max
        band_weights = [1.0 + 0.5 * l for l in range(1, l_max + 1)]
    loss = torch.zeros((), device=delta_rest.device, dtype=delta_rest.dtype)
    idx = 0
    for l in range(1, l_max + 1):
        n_b = (l + 1) ** 2 - (l**2)
        w = float(band_weights[l - 1]) if l - 1 < len(band_weights) else float(band_weights[-1])
        chunk = delta_rest[:, idx : idx + n_b, :]
        loss = loss + w * chunk.pow(2).mean()
        idx += n_b
    return loss
