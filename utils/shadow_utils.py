"""Heuristic drop-shadow Gaussians for compositing.

A `ShadowGaussians` object is a minimal stand-in for a `DynamicGaussianModel` that the
existing `utils.transform_utils_torch.render` loop can consume when called with
`static=True`. It carries a flat, dark, semi-opaque oval cluster of Gaussians in
world-space coordinates so a composited foreground object visually casts a shadow on the
support plane below it.

Build one with `build_shadow_under_object(...)` and pass it as a third entry in the
`gaussians` list (alongside the BG and FG).
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from utils.sh_utils import RGB2SH


class ShadowGaussians:
    """Minimal Gaussian set for static drop-shadow rendering.

    The render loop's `static=True` branch reads `get_xyz`, `get_opacity`,
    `get_scaling`, `get_rotation`, `get_features`. Storage is in *raw* (pre-activation)
    space; properties apply the standard activations.
    """

    def __init__(self, xyz, scales_act, rotations, opacity_act, sh_features):
        # Storage convention mirrors the real GaussianModel: scales in log-space,
        # rotations as raw quaternions, opacity in logit-space.
        eps = 1e-4
        opa_clipped = opacity_act.clamp(eps, 1.0 - eps)
        self._xyz = xyz                                           # (N, 3) world coords
        self._scaling = torch.log(scales_act.clamp(min=1e-8))     # (N, 3) raw (log)
        self._rotation = rotations                                # (N, 4) quaternion
        self._opacity = torch.log(opa_clipped / (1.0 - opa_clipped))  # (N, 1) logit
        self._features_dc = sh_features[:, 0:1, :]                # (N, 1, 3)
        self._features_rest = sh_features[:, 1:, :]               # (N, 15, 3)
        self.active_sh_degree = 3

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @staticmethod
    def scaling_activation(x):
        return torch.exp(x)

    @staticmethod
    def opacity_activation(x):
        return torch.sigmoid(x)

    @staticmethod
    def rotation_activation(x):
        return F.normalize(x, dim=-1)


def build_shadow_under_object(
    *,
    fg_world_xyz: torch.Tensor,
    down_axis: torch.Tensor = torch.tensor([0.0, 1.0, 0.0]),
    drop_distance: float = 0.05,
    n_points: int = 2000,
    half_extent: tuple = (0.45, 0.45),
    thickness: float = 0.02,
    rgb_color: tuple = (0.04, 0.03, 0.02),
    opacity: float = 0.55,
    falloff: float = 2.0,
    fg_weights: torch.Tensor = None,
    bottom_percentile: float = 0.85,
    gaussian_size_mult: float = 4.0,
    plane_point: torch.Tensor = None,
    auto_size_factor: float = None,
    device: str = "cuda",
) -> ShadowGaussians:
    """Build a flat oval of dark, semi-opaque Gaussians directly below an object.

    There are two placement modes:

    * **Plane-projection (preferred, used when `plane_point` is supplied)**: place the
      shadow centre at the orthogonal projection of the FG (weighted) centroid onto the
      support plane defined by `(plane_point, down_axis)`. The disk lies in the plane
      perpendicular to `down_axis`, so the shadow is always *directly below* the
      object regardless of how oblique `down_axis` is. `drop_distance` is then used
      as a small offset *into* the surface to avoid z-fighting with the BG.

    * **Drop-along-axis (legacy fallback, used when `plane_point` is None)**: push the
      shadow centre along `down_axis` from the FG centroid by
      `(lowest_offset + drop_distance)`. This gets the placement wrong for oblique
      down axes, where the disk ends up offset sideways and largely occluded by the
      cutting board / object body — left in for backward compatibility only.

    Args:
        fg_world_xyz: (M, 3) world coords of the foreground object's masked Gaussians,
            already transformed (post motion_bias / scale_bias / rotation_bias).
        down_axis: unit vector pointing from the object towards the support surface in
            world frame. Used both for disk orientation and (in the legacy mode) for
            shadow placement.
        plane_point: optional (3,) point on the support plane (e.g. mean of nearby BG
            Gaussians from `estimate_support_plane_normal`). When provided, the shadow
            is placed via orthogonal projection of the FG centroid onto this plane.
        drop_distance: world units to push the shadow into the surface (along
            `down_axis`) to avoid z-fighting / depth ambiguity. Small positive values
            (~0.005..0.02) are typical when `plane_point` is supplied.
        auto_size_factor: optional. When set, `half_extent` is replaced by the FG's
            in-plane footprint radius (90th percentile of FG points projected onto the
            plane) multiplied by this factor. Disabled (uses the explicit half_extent)
            when None.
        n_points: total Gaussians in the shadow plate.
        half_extent: (a, b) half-axes of the shadow oval in the support plane (world
            units along the two axes orthogonal to `down_axis`).
        thickness: out-of-plane thickness (world units) — keep small so the shadow
            is a flat plate rather than a sphere.
        rgb_color: dark color in [0, 1] linear RGB.
        opacity: peak per-Gaussian opacity at the disk centre.
        falloff: exponent of the radial opacity falloff (1 = linear, 2 = quadratic).
        fg_weights: per-FG-Gaussian weights (e.g. opacities) used for the centroid and
            footprint percentile computation. None ⇒ uniform weights.
        bottom_percentile: only used in the legacy drop-along-axis mode.
        gaussian_size_mult: per-Gaussian softness multiplier.

    Returns:
        ShadowGaussians instance ready to be passed into the existing render loop with
        `static=True, seg=False, motion_bias=zeros, rotation_bias=zeros, scales_bias=1`.
    """
    fg_world_xyz = fg_world_xyz.to(device)
    down_axis = down_axis.to(device).float()
    down_axis = down_axis / down_axis.norm()

    # Object centroid in world frame, weighted by opacity if provided so the centre
    # tracks the *visible* mass rather than dragged-out stray Gaussians.
    if fg_weights is not None:
        w = fg_weights.to(device).float().clamp(min=0).reshape(-1)
        if w.sum() > 0:
            centre = (fg_world_xyz * w.unsqueeze(1)).sum(0) / w.sum()
        else:
            centre = fg_world_xyz.mean(dim=0)
            w = None
    else:
        centre = fg_world_xyz.mean(dim=0)
        w = None

    if plane_point is not None:
        # Plane-projection placement: orthogonally project the FG centroid onto the
        # support plane (defined by `plane_point` with normal -down_axis), then nudge
        # `drop_distance` into the surface (along down_axis) to avoid z-fighting.
        p = plane_point.to(device).float().reshape(3)
        signed = ((centre - p) * down_axis).sum()       # how far along +down centre is from plane
        floor_centre = centre - signed * down_axis + float(drop_distance) * down_axis
    else:
        # Legacy drop-along-axis placement (kept for backward compat; produces sideways
        # offsets when down_axis isn't world-axis-aligned).
        proj = (fg_world_xyz - centre) @ down_axis     # (M,) — larger = farther down
        if w is not None:
            order = torch.argsort(proj)
            proj_sorted = proj[order]
            ws = w[order]
            cw = torch.cumsum(ws, dim=0)
            target = bottom_percentile * cw[-1]
            idx = int(torch.searchsorted(cw, target).item())
            idx = max(0, min(idx, proj_sorted.shape[0] - 1))
            lowest_offset = proj_sorted[idx]
        else:
            k = max(1, int((1.0 - float(bottom_percentile)) * proj.shape[0]))
            lowest_offset = torch.kthvalue(proj, proj.shape[0] - k + 1).values
        floor_centre = centre + (lowest_offset + drop_distance) * down_axis

    # Build orthonormal frame: down_axis + two perpendicular axes (axis_a, axis_b).
    helper = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=down_axis.dtype)
    if torch.abs((down_axis * helper).sum()) > 0.95:
        helper = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=down_axis.dtype)
    axis_a = F.normalize(torch.cross(down_axis, helper, dim=0), dim=0)
    axis_b = F.normalize(torch.cross(down_axis, axis_a, dim=0), dim=0)

    # Optionally auto-size the disk to match the FG's in-plane footprint.
    # FG point clouds typically have a *very* long opacity-weighted tail of stray
    # Gaussians far from the visible mass, so we use a robust 25th percentile of
    # in-plane radius from the high-opacity bulk and rely on `auto_size_factor`
    # (typically 1.5..2.5) to expand to a usable disk size.
    if auto_size_factor is not None:
        proj_a = (fg_world_xyz - floor_centre.unsqueeze(0)) @ axis_a    # (M,)
        proj_b = (fg_world_xyz - floor_centre.unsqueeze(0)) @ axis_b    # (M,)
        rad = torch.sqrt(proj_a ** 2 + proj_b ** 2)

        if w is not None:
            # Keep only the top 25% by opacity weight to suppress stray Gaussians.
            thr = float(torch.quantile(w, 0.75).item())
            keep = w >= max(thr, 1e-3)
            rad_keep = rad[keep] if keep.any() else rad
            footprint = float(torch.quantile(rad_keep, 0.5).item())
        else:
            footprint = float(torch.quantile(rad, 0.5).item())
        a = b = max(0.05, footprint * float(auto_size_factor))
    else:
        a, b = float(half_extent[0]), float(half_extent[1])

    # Sample disk via stratified jittered grid for even coverage.
    side = int(math.ceil(math.sqrt(n_points)))
    u = torch.linspace(-1.0, 1.0, side, device=device)
    gx, gy = torch.meshgrid(u, u, indexing="ij")
    gx = gx.flatten() + (torch.rand_like(gx.flatten()) - 0.5) * (2.0 / side)
    gy = gy.flatten() + (torch.rand_like(gy.flatten()) - 0.5) * (2.0 / side)
    rsq = gx ** 2 + gy ** 2
    inside = rsq <= 1.0
    gx, gy, rsq = gx[inside], gy[inside], rsq[inside]
    # Trim to requested count.
    if gx.shape[0] > n_points:
        idx = torch.randperm(gx.shape[0], device=device)[:n_points]
        gx, gy, rsq = gx[idx], gy[idx], rsq[idx]
    n = gx.shape[0]

    offsets = (gx * a).unsqueeze(1) * axis_a + (gy * b).unsqueeze(1) * axis_b
    xyz = floor_centre.unsqueeze(0) + offsets

    # Per-Gaussian scale: in-plane scaled to disk density, out-of-plane = thickness.
    # gaussian_size_mult controls how much each Gaussian overlaps its neighbours; larger
    # values produce a denser, more opaque-looking shadow plate at the cost of a softer edge.
    in_plane_scale = max(a, b) / (n ** 0.5) * float(gaussian_size_mult)
    scales = torch.zeros((n, 3), device=device)
    # Place the flat dimension along the local "y" of a Gaussian quaternion frame
    # below; we'll rotate every Gaussian so its short axis lines up with down_axis.
    scales[:, 0] = in_plane_scale
    scales[:, 1] = thickness
    scales[:, 2] = in_plane_scale

    # Rotation: quaternion that maps the canonical Y axis to `down_axis`.
    canonical = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=down_axis.dtype)
    cos_t = (canonical * down_axis).sum().clamp(-1.0, 1.0)
    if cos_t > 1.0 - 1e-6:
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=down_axis.dtype)  # identity
    elif cos_t < -1.0 + 1e-6:
        q = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=down_axis.dtype)  # 180° flip
    else:
        axis = F.normalize(torch.cross(canonical, down_axis, dim=0), dim=0)
        half = math.acos(float(cos_t)) * 0.5
        s = math.sin(half)
        q = torch.stack([torch.tensor(math.cos(half), device=device, dtype=down_axis.dtype),
                         axis[0] * s, axis[1] * s, axis[2] * s])
    rotations = q.unsqueeze(0).repeat(n, 1)

    # Radial opacity falloff (peak at centre, 0 at edge).
    radial = (1.0 - rsq.clamp(0, 1)) ** float(falloff)
    opa = (radial * float(opacity)).clamp(1e-4, 1.0 - 1e-4).unsqueeze(1)

    # SH features: degree 0 only — flat dark color.
    sh_dc = RGB2SH(torch.tensor(rgb_color, device=device, dtype=torch.float32)).unsqueeze(0).repeat(n, 1)
    sh_features = torch.zeros((n, 16, 3), device=device, dtype=torch.float32)
    sh_features[:, 0, :] = sh_dc

    return ShadowGaussians(xyz.float(), scales.float(), rotations.float(), opa.float(), sh_features)
