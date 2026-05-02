#!/usr/bin/env bash
set -euo pipefail

# Example stress-test commands for two scenes in this repo.
# Adjust iterations / amplify / feather for your machine.

ROOT="/home/ubuntu/new_sa4d/sa4d"
PY="${CONDA_PREFIX:-/home/ubuntu/miniconda3/envs/sa4d}/bin/python"

split_cookie() {
  local MODEL="${ROOT}/output/hypernerf/split-cookie"
  local SRC="${ROOT}/data/hypernerf/split-cookie"
  local PLY="${MODEL}/point_cloud/iteration_14000/clean_chocolate_Bigger.ply"
  local MASK="${MODEL}/segment_results/composite_inserted_choc_Bigger.pt"
  local OUT_PLY="${MODEL}/point_cloud/iteration_14000/harmonized_split_cookie_bigger_shadow.ply"

  "${PY}" -m pipeline.run_harmonize \
    --model_path "${MODEL}" \
    --source_path "${SRC}" \
    --mask_path "${MASK}" \
    --ply_path "${PLY}" \
    --output_ply "${OUT_PLY}" \
    --iteration 14000 \
    --harmonizer whitebox \
    --mask_feather 6 \
    --amplify 1.2 \
    --num_iterations 2000 \
    --lr 1e-3 --lr_dc 5e-4 --lr_rest 3e-3 \
    --reg_weight 0.02 \
    --shadow_mode learned --shadow_n 2048 --shadow_lr 5e-3 \
    --shadow_reg_weight 0.01 --shadow_outside_weight 0.05
}

torchocolate() {
  local MODEL="${ROOT}/output/hypernerf/torchocolate"
  local SRC="${ROOT}/data/hypernerf/torchocolate"
  # NOTE: you must pick a PLY + mask pair with matching Gaussian counts.
  # This is a template; fill in correct paths from your segment_results/ + point_cloud/.
  local PLY="${MODEL}/point_cloud/iteration_14000/scene_point_cloud.ply"
  local MASK="${MODEL}/segment_results/torchocolate.pt"
  local OUT_PLY="${MODEL}/point_cloud/iteration_14000/harmonized_torchocolate_shadow.ply"

  "${PY}" -m pipeline.run_harmonize \
    --model_path "${MODEL}" \
    --source_path "${SRC}" \
    --mask_path "${MASK}" \
    --ply_path "${PLY}" \
    --output_ply "${OUT_PLY}" \
    --iteration 14000 \
    --harmonizer whitebox \
    --mask_feather 6 \
    --amplify 1.2 \
    --num_iterations 2000 \
    --lr 1e-3 --lr_dc 5e-4 --lr_rest 3e-3 \
    --reg_weight 0.02 \
    --shadow_mode learned --shadow_n 2048 --shadow_lr 5e-3 \
    --shadow_reg_weight 0.01 --shadow_outside_weight 0.05
}

case "${1:-}" in
  split-cookie) split_cookie ;;
  torchocolate) torchocolate ;;
  *)
    echo "usage: $0 {split-cookie|torchocolate}"
    exit 2
    ;;
esac
