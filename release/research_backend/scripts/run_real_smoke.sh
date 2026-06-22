#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DIGIMON_ROOT:?Set DIGIMON_ROOT to a compatible DIGIMON checkout}"
: "${ARC_FUSE_API_KEY:?Set ARC_FUSE_API_KEY}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-datasets/Popqa}"

"${PYTHON_BIN}" "${ROOT}/arc_fuse_main.py" \
  --digimon-root "${DIGIMON_ROOT}" \
  -opt "${ROOT}/research_backend/configs/arc_fuse.yaml" \
  -graph_method hipporag \
  -text_method bm25 \
  -dataset_name "${DATASET}" \
  --eval_limit 1
