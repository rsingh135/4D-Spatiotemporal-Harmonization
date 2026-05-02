"""
Summarize harmonization runs into a single JSON for presentation/evidence.

This scans a sweep directory (or a model_path) for files written by pipeline/run_harmonize.py:
  - *_losses.pt (contains losses + args)
  - *loss_curve.png
  - delta_sh.pt (optional, if you point at a run output folder)

It produces a summary JSON with final/avg losses and the key ablation flags.
"""

import os
import json
import argparse
from typing import Any, Dict, List

import torch


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _summarize_loss_file(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    losses = data.get("losses", [])
    args = data.get("args", {})
    losses_f = [_safe_float(v) for v in losses]

    tail_n = min(50, len(losses_f)) if losses_f else 0
    tail_avg = sum(losses_f[-tail_n:]) / max(1, tail_n) if tail_n else float("nan")

    return {
        "losses_pt": path,
        "n_steps": int(len(losses_f)),
        "loss_first": losses_f[0] if losses_f else float("nan"),
        "loss_last": losses_f[-1] if losses_f else float("nan"),
        "loss_tail50_avg": tail_avg,
        "args": {
            # high-signal knobs
            "harmonizer": args.get("harmonizer"),
            "amplify": args.get("amplify"),
            "mask_feather": args.get("mask_feather"),
            "mask_core_erode_px": args.get("mask_core_erode_px", 0),
            "mask_boundary_weight": args.get("mask_boundary_weight", 0.25),
            "mask_weight_power": args.get("mask_weight_power", 1.0),
            "use_lpips": args.get("use_lpips", False),
            "shadow_mode": args.get("shadow_mode", "off"),
            "lr": args.get("lr"),
            "lr_dc": args.get("lr_dc"),
            "lr_rest": args.get("lr_rest"),
            "reg_weight": args.get("reg_weight"),
            "num_iterations": args.get("num_iterations"),
            "ply_path": args.get("ply_path"),
            "mask_path": args.get("mask_path"),
            "output_ply": args.get("output_ply"),
        },
    }


def _collect_losses(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("_losses.pt"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize harmonize sweep runs")
    p.add_argument("--root", type=str, required=True, help="Directory to scan for *_losses.pt")
    p.add_argument("--out_json", type=str, required=True, help="Where to write the summary JSON")
    args = p.parse_args()

    loss_files = _collect_losses(args.root)
    rows = []
    for lf in loss_files:
        try:
            rows.append(_summarize_loss_file(lf))
        except Exception as e:
            rows.append({"losses_pt": lf, "error": str(e)})

    payload = {
        "root": os.path.abspath(args.root),
        "n_runs": int(len(rows)),
        "runs": rows,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[summarize] wrote {args.out_json} with {len(rows)} runs")


if __name__ == "__main__":
    main()

