#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DIGIMON_ROOT:?Set DIGIMON_ROOT to a compatible DIGIMON checkout}"
: "${ARC_FUSE_API_KEY:?Set ARC_FUSE_API_KEY}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-datasets/musique}"
GRAPH="${GRAPH:-hipporag}"
TEXT="${TEXT:-bm25}"
LIMIT="${LIMIT:-200}"

CONFIGS=(
  arc_fuse_abl_simple.yaml
  arc_fuse_abl_routing.yaml
  arc_fuse_abl_regen.yaml
  arc_fuse_abl_critic.yaml
  arc_fuse_abl_no_commendor.yaml
  arc_fuse_abl_single_agent.yaml
  arc_fuse.yaml
)

for config in "${CONFIGS[@]}"; do
  echo "ARC-Fuse ablation=${config} dataset=${DATASET}"
  "${PYTHON_BIN}" "${ROOT}/arc_fuse_main.py" \
    --digimon-root "${DIGIMON_ROOT}" \
    -opt "${ROOT}/research_backend/configs/${config}" \
    -graph_method "${GRAPH}" \
    -text_method "${TEXT}" \
    -dataset_name "${DATASET}" \
    --eval_limit "${LIMIT}"
done
