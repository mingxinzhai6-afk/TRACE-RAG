#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${ROOT}/run_demo.py" \
  --config "${ROOT}/configs/arc_fuse.example.json" \
  --corpus "${ROOT}/examples/corpus.jsonl" \
  --graph "${ROOT}/examples/graph.jsonl" \
  --questions "${ROOT}/examples/questions.jsonl" \
  --offline \
  --output "${ROOT}/output/demo_results.jsonl"
