#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DIGIMON_ROOT:?Set DIGIMON_ROOT to a compatible DIGIMON checkout}"
: "${ARC_FUSE_API_KEY:?Set ARC_FUSE_API_KEY}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LIMIT="${LIMIT:-200}"
DATASETS="${DATASETS:-datasets/Popqa datasets/musique}"
CONFIG="${ROOT}/research_backend/configs/arc_fuse.yaml"

for dataset in ${DATASETS}; do
  for graph in hipporag tog raptor; do
    for text in bm25 vdb; do
      echo "ARC-Fuse dataset=${dataset} graph=${graph} text=${text}"
      "${PYTHON_BIN}" "${ROOT}/arc_fuse_main.py" \
        --digimon-root "${DIGIMON_ROOT}" \
        -opt "${CONFIG}" \
        -graph_method "${graph}" \
        -text_method "${text}" \
        -dataset_name "${dataset}" \
        --eval_limit "${LIMIT}"
    done
  done
done
