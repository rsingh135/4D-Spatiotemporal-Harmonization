#!/usr/bin/env bash
set -euo pipefail

# Package harmonization evidence into a single folder:
# - sweep summary JSON (loss curves + key args)
# - copies of *_loss_curve.png (if present)
# - optionally results.json from metrics.py runs (if present under model_path)
#
# Usage:
#   cd /home/ubuntu/new_sa4d/sa4d
#   bash scripts/package_presentation_results.sh \
#     --sweep_root output/hypernerf/split-cookie/harmonize_sweeps/<RUN_TAG> \
#     --out_dir output/presentation/<TAG>

ROOT="/home/ubuntu/new_sa4d/sa4d"
PY="${CONDA_PREFIX:-/home/ubuntu/miniconda3/envs/sa4d}/bin/python"

SWEEP_ROOT=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep_root) SWEEP_ROOT="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

if [[ -z "${SWEEP_ROOT}" || -z "${OUT_DIR}" ]]; then
  echo "usage: $0 --sweep_root <dir> --out_dir <dir>"
  exit 2
fi

mkdir -p "${OUT_DIR}"

SUMMARY_JSON="${OUT_DIR}/harmonize_summary.json"
"${PY}" -m pipeline.summarize_harmonize_runs --root "${SWEEP_ROOT}" --out_json "${SUMMARY_JSON}"

# Copy loss curve pngs (if any)
mkdir -p "${OUT_DIR}/loss_curves"
shopt -s globstar nullglob
for f in "${SWEEP_ROOT}"/**/*loss_curve.png; do
  cp -f "${f}" "${OUT_DIR}/loss_curves/$(basename "${f}")"
done

# Copy harmonize diffs (if any)
mkdir -p "${OUT_DIR}/diff_strips"
for f in "${SWEEP_ROOT}"/**/*.png; do
  bn="$(basename "${f}")"
  if [[ "${bn}" == view*frame*.png ]]; then
    cp -f "${f}" "${OUT_DIR}/diff_strips/${bn}"
  fi
done

echo "[package] wrote: ${OUT_DIR}"
echo "[package] summary: ${SUMMARY_JSON}"

