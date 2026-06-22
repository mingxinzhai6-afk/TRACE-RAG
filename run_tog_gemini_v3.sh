#!/bin/bash
# Run the third Gemini ToG prompt variant as the normal Gemini ToG result.
#
# Outputs:
# - output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/

set -euo pipefail

cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
mkdir -p logs
PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"

if [ "${ALLOW_PARALLEL:-0}" != "1" ]; then
  active_jobs="$(ps -ef | grep -E 'run_20_main|run_popqa_reruns|run_ablation|main.py|newg_main.py' | grep -v grep || true)"
  if [ -n "$active_jobs" ]; then
    echo "Another experiment appears to be running:"
    echo "$active_jobs"
    echo "Exit without starting ToG v3. Re-run with ALLOW_PARALLEL=1 to override."
    exit 1
  fi
fi

set_model() {
  "${PYTHON}" - "$1" <<'PY'
from pathlib import Path
import re
import sys

model = sys.argv[1]
p = Path("Option/Config2.yaml")
text = p.read_text()
new_text, n = re.subn(
    r'(?m)^(\s*model:\s*)"[^\"]+"',
    lambda m: f'{m.group(1)}"{model}"',
    text,
    count=1,
)
if n != 1:
    raise SystemExit("failed to update llm.model in Option/Config2.yaml")
p.write_text(new_text)
PY
  grep -n 'model:' Option/Config2.yaml | head -2
}

summarize_score_file() {
  local label="$1"
  local path="$2"
  python3 - "$label" "$path" <<'PY'
import json
import os
import sys

label, path = sys.argv[1], sys.argv[2]
metrics = ["accuracy", "em", "precision", "recall", "f1"]
if not os.path.exists(path):
    print(f"{label}\tMISSING\t{path}")
    raise SystemExit(0)
rows = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
vals = [
    sum(float(row.get(metric, 0) or 0) for row in rows) / len(rows)
    for metric in metrics
]
print(
    f"{label}\tn={len(rows)}\t"
    + "\t".join(f"{metric}={value * 100:.2f}%" for metric, value in zip(metrics, vals))
)
PY
}

set_model gemini-2.5-flash-lite

target_dir="output/datasets/Popqa/ToG_gemini-2.5-flash-lite"
if [ -e "$target_dir" ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  archive_dir=".tog_gemini_rerun_archived_${ts}"
  mkdir -p "$archive_dir/$(dirname "$target_dir")"
  mv "$target_dir" "$archive_dir/$target_dir"
  echo "archived_old_tog=$archive_dir/$target_dir"
fi

echo "=== $(date '+%F %T') run PopQA Gemini ToG ==="
"${PYTHON}" main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200 \
  > logs/rerun_popqa_tog_gemini_v3.log 2>&1

summarize_score_file \
  "PopQA gemini ToG" \
  output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.score.json

echo "=== Gemini ToG rerun finished ==="
