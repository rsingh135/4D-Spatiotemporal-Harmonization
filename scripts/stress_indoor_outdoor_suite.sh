#!/usr/bin/env bash
set -euo pipefail

# Indoor↔Outdoor stress suite for harmonization.
#
# What it does:
#  1) (optional) postprocess an existing mask_table (.pt) to remove islands/leakage
#  2) generate several object-only SH mismatches in 3D (indoor_outdoor, harsh_shadow, full_mismatch)
#  3) run harmonization across a small ablation grid and save:
#       - harmonized .ply
#       - cached targets (per mismatch)
#       - per-view diff strips
#       - loss curves (.pt + .png) from run_harmonize.py
#
# Usage:
#   cd /home/ubuntu/new_sa4d/sa4d
#   bash scripts/stress_indoor_outdoor_suite.sh split-cookie
#   bash scripts/stress_indoor_outdoor_suite.sh torchocolate
#
# Notes:
# - Requires that MODEL/SRC exist and that MASK+PLY have matching Gaussian counts.

ROOT="/home/ubuntu/new_sa4d/sa4d"
PY="${CONDA_PREFIX:-/home/ubuntu/miniconda3/envs/sa4d}/bin/python"

scene_split_cookie() {
  local MODEL="${ROOT}/output/hypernerf/split-cookie"
  local SRC="${ROOT}/data/hypernerf/split-cookie"
  local PLY="${MODEL}/point_cloud/iteration_14000/clean_chocolate_Bigger.ply"
  local MASK="${MODEL}/segment_results/composite_inserted_choc_Bigger.pt"
  echo "${MODEL}|${SRC}|${PLY}|${MASK}"
}

scene_torchocolate() {
  local MODEL="${ROOT}/output/hypernerf/torchocolate"
  local SRC="${ROOT}/data/hypernerf/torchocolate"
  local PLY="${MODEL}/point_cloud/iteration_14000/scene_point_cloud.ply"
  local MASK="${MODEL}/segment_results/torchocolate.pt"
  echo "${MODEL}|${SRC}|${PLY}|${MASK}"
}

pick_scene() {
  case "${1:-}" in
    split-cookie) scene_split_cookie ;;
    torchocolate) scene_torchocolate ;;
    *)
      echo "usage: $0 {split-cookie|torchocolate}"
      exit 2
      ;;
  esac
}

IFS='|' read -r MODEL SRC PLY MASK < <(pick_scene "${1:-}")

ITER=14000
CFG="arguments/hypernerf/default.py"

RUN_TAG="indoor_outdoor_suite_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${MODEL}/harmonize_sweeps/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo "[suite] MODEL=${MODEL}"
echo "[suite] SRC=${SRC}"
echo "[suite] PLY=${PLY}"
echo "[suite] MASK=${MASK}"
echo "[suite] OUT_DIR=${OUT_DIR}"

# --- Step 1: mask postprocess (optional but recommended) ---
MASK_CLEAN="${OUT_DIR}/mask_clean.pt"
"${PY}" -m pipeline.postprocess_mask_table \
  --model_path "${MODEL}" \
  --source_path "${SRC}" \
  --mask_path "${MASK}" \
  --out_path "${MASK_CLEAN}" \
  --iteration "${ITER}" \
  --configs "${CFG}" \
  --ply_path "${PLY}" \
  --min_frac 0.15 \
  --knn_k 16 \
  --knn_radius 0.06

# --- Step 2: make several 3D-relevant mismatch PLYs ---
declare -a EFFECTS=("indoor_outdoor" "harsh_shadow" "full_mismatch")
declare -a STRENGTHS=("1.0" "1.5" "2.0")

for effect in "${EFFECTS[@]}"; do
  for strength in "${STRENGTHS[@]}"; do
    OUT_PLY="${OUT_DIR}/mismatch_${effect}_s${strength}.ply"
    "${PY}" -m pipeline.create_test_scene \
      --model_path "${MODEL}" \
      --source_path "${SRC}" \
      --mask_path "${MASK_CLEAN}" \
      --ply_path "${PLY}" \
      --output_ply "${OUT_PLY}" \
      --effect "${effect}" \
      --strength "${strength}" \
      --target foreground \
      --iteration "${ITER}" \
      --configs "${CFG}"
  done
done

# --- Step 3: harmonize across a compact ablation grid ---
declare -a HARMONIZERS=("whitebox" "pctnet")
declare -a FEATHERS=("0" "6")

# New: core/boundary weights (reduce halos)
CORE_ERODE=2
BOUNDARY_W=0.25
WEIGHT_POW=1.5

for mismatch_ply in "${OUT_DIR}"/mismatch_*.ply; do
  base="$(basename "${mismatch_ply%.ply}")"
  for h in "${HARMONIZERS[@]}"; do
    for feather in "${FEATHERS[@]}"; do
      RUN="${base}__h${h}__f${feather}"
      OUT_PLY="${OUT_DIR}/harmonized_${RUN}.ply"

      "${PY}" -m pipeline.run_harmonize \
        --model_path "${MODEL}" \
        --source_path "${SRC}" \
        --mask_path "${MASK_CLEAN}" \
        --ply_path "${mismatch_ply}" \
        --output_ply "${OUT_PLY}" \
        --iteration "${ITER}" \
        --configs "${CFG}" \
        --harmonizer "${h}" \
        --no_diffs \
        --mask_feather "${feather}" \
        --mask_core_erode_px "${CORE_ERODE}" \
        --mask_boundary_weight "${BOUNDARY_W}" \
        --mask_weight_power "${WEIGHT_POW}" \
        --amplify 1.2 \
        --num_iterations 1500 \
        --lr 1e-3 --lr_dc 5e-4 --lr_rest 3e-3 \
        --reg_weight 0.02 \
        --shadow_mode learned --shadow_n 2048 --shadow_lr 5e-3 \
        --shadow_reg_weight 0.01 --shadow_outside_weight 0.05
    done
  done
done

echo "[suite] Done. Outputs in: ${OUT_DIR}"

