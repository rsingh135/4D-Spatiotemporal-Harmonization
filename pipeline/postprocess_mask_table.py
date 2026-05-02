"""
Postprocess a per-Gaussian mask table (.pt) to reduce leakage / islands.

This is meant to improve downstream harmonization stability by cleaning the *3D*
Gaussian selection before it is projected to 2D via render_mask().

What it does
------------
Given an input mask table:
  mask_table: [T, N] bool
  time_map:   [T]

We build a union mask across time and then apply:
  1) Temporal persistence filter:
       keep gaussians that are active in at least min_frac of frames
  2) Largest connected component filter (in XYZ):
       build a kNN graph over kept gaussians and keep only the biggest component

The resulting mask_table is written to --out_path, preserving metadata keys.
"""

import os
import sys
import argparse
from typing import Dict, Tuple

import torch

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


def _union_find_largest_component(edges_src: torch.Tensor, edges_dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Union-find connected components and return a boolean mask selecting the largest component.
    edges_src/dst are 1D int64 tensors on CPU with values in [0, num_nodes).
    """
    parent = list(range(num_nodes))
    size = [1] * num_nodes

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in zip(edges_src.tolist(), edges_dst.tolist()):
        union(int(a), int(b))

    # compress + count
    comp_sizes: Dict[int, int] = {}
    for i in range(num_nodes):
        r = find(i)
        comp_sizes[r] = comp_sizes.get(r, 0) + 1
    if not comp_sizes:
        return torch.zeros((num_nodes,), dtype=torch.bool)
    largest_root = max(comp_sizes, key=lambda r: comp_sizes[r])
    keep = torch.zeros((num_nodes,), dtype=torch.bool)
    for i in range(num_nodes):
        keep[i] = find(i) == largest_root
    return keep


def _build_knn_edges(xyz: torch.Tensor, k: int, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build directed kNN edges (i -> neighbor) within a distance threshold.

    Args:
        xyz:    [M,3] float tensor on CUDA
        k:      number of neighbors (excluding self)
        radius: include edges whose euclidean distance <= radius

    Returns:
        (src, dst): int64 CPU tensors of equal length
    """
    import pytorch3d.ops

    with torch.no_grad():
        # knn_points expects [B,M,3]
        knn = pytorch3d.ops.knn_points(
            xyz[None, ...], xyz[None, ...], K=int(k) + 1, return_nn=False
        )
        idx = knn.idx.squeeze(0)  # [M, k+1]
        dists = knn.dists.squeeze(0)  # [M, k+1] squared distances

        # drop self neighbor at [:,0]
        idx = idx[:, 1:]
        d = dists[:, 1:].clamp_min(0).sqrt()

        # keep edges under radius
        keep = d <= float(radius)
        src = torch.arange(xyz.shape[0], device=xyz.device)[:, None].expand_as(idx)
        src = src[keep].detach().cpu().to(torch.int64)
        dst = idx[keep].detach().cpu().to(torch.int64)
        return src, dst


def postprocess_mask_table(
    mask_data: Dict,
    xyz: torch.Tensor,
    *,
    min_frac: float,
    knn_k: int,
    knn_radius: float,
) -> Dict:
    """
    Returns a new mask_data dict with a cleaned mask_table.
    """
    if "mask_table" not in mask_data or "time_map" not in mask_data:
        raise ValueError("mask_data must contain 'mask_table' and 'time_map'")

    mask_table = mask_data["mask_table"]
    if mask_table.dtype != torch.bool:
        mask_table = mask_table.bool()

    if mask_table.ndim != 2:
        raise ValueError(f"mask_table must be [T,N], got shape {tuple(mask_table.shape)}")
    T, N = mask_table.shape
    if xyz.shape[0] != N:
        raise ValueError(f"Gaussian count mismatch: mask N={N}, xyz N={int(xyz.shape[0])}")

    # 1) temporal persistence
    frac = mask_table.float().mean(dim=0)  # [N]
    keep_temporal = frac >= float(min_frac)

    # 2) LCC in XYZ of union mask after temporal filter
    union = mask_table.any(dim=0)
    keep0 = union & keep_temporal
    if keep0.sum().item() == 0:
        # Nothing to keep: return empty masks but preserve metadata
        out = dict(mask_data)
        out["mask_table"] = torch.zeros_like(mask_table, dtype=torch.bool)
        out["postprocess"] = {
            "min_frac": float(min_frac),
            "knn_k": int(knn_k),
            "knn_radius": float(knn_radius),
            "kept_temporal": int(keep_temporal.sum().item()),
            "kept_lcc": 0,
        }
        return out

    sel_idx = torch.nonzero(keep0, as_tuple=False).squeeze(1)
    xyz_sel = xyz[sel_idx].contiguous()

    src, dst = _build_knn_edges(xyz_sel, k=knn_k, radius=knn_radius)
    if src.numel() == 0:
        keep_sel = torch.ones((xyz_sel.shape[0],), dtype=torch.bool)
    else:
        keep_sel = _union_find_largest_component(src, dst, num_nodes=int(xyz_sel.shape[0]))

    keep_lcc = torch.zeros((N,), dtype=torch.bool, device=mask_table.device)
    keep_lcc[sel_idx] = keep_sel.to(device=mask_table.device)

    cleaned = mask_table & keep_lcc[None, :]

    out = dict(mask_data)
    out["mask_table"] = cleaned.bool()
    out["postprocess"] = {
        "min_frac": float(min_frac),
        "knn_k": int(knn_k),
        "knn_radius": float(knn_radius),
        "kept_temporal": int(keep_temporal.sum().item()),
        "kept_lcc": int(keep_lcc.sum().item()),
        "T": int(T),
        "N": int(N),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Postprocess a per-Gaussian mask table (.pt)")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--source_path", type=str, required=True)
    p.add_argument("--mask_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--configs", type=str, default=None)
    p.add_argument("--ply_path", type=str, default=None, help="Override which .ply to load (must match mask ordering)")

    p.add_argument("--min_frac", type=float, default=0.15,
                   help="Keep gaussians active in at least this fraction of frames (0-1).")
    p.add_argument("--knn_k", type=int, default=16, help="k for kNN graph (excluding self).")
    p.add_argument("--knn_radius", type=float, default=0.06,
                   help="Edge radius threshold in XYZ units for kNN connectivity.")
    args = p.parse_args()

    from pipeline.data_loading import load_scene, load_mask_table

    gaussians, scene, pipe, bg = load_scene(
        args.model_path, args.source_path, iteration=args.iteration, configs=args.configs
    )
    _ = scene, pipe, bg

    if args.ply_path is not None:
        gaussians.load_ply(args.ply_path)
        if hasattr(gaussians, "_deformation_table"):
            n_xyz = gaussians._xyz.shape[0]
            if (not torch.is_tensor(gaussians._deformation_table)) or (gaussians._deformation_table.shape[0] != n_xyz):
                gaussians._deformation_table = torch.ones((n_xyz,), device="cuda", dtype=torch.bool)

    mask_data = load_mask_table(args.mask_path)
    xyz = gaussians._xyz  # [N,3] on CUDA

    out = postprocess_mask_table(
        mask_data, xyz,
        min_frac=args.min_frac,
        knn_k=args.knn_k,
        knn_radius=args.knn_radius,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(out, args.out_path)

    before_active = int(mask_data["mask_table"].float().mean().item() * mask_data["mask_table"].numel())
    after_active = int(out["mask_table"].float().mean().item() * out["mask_table"].numel())

    print(f"[mask_post] wrote: {args.out_path}")
    print(f"[mask_post] union before: {int(mask_data['mask_table'].any(dim=0).sum().item())} gaussians")
    print(f"[mask_post] union after:  {int(out['mask_table'].any(dim=0).sum().item())} gaussians")
    print(f"[mask_post] active entries approx before={before_active} after={after_active}")
    print(f"[mask_post] params: min_frac={args.min_frac} knn_k={args.knn_k} knn_radius={args.knn_radius}")


if __name__ == "__main__":
    main()

