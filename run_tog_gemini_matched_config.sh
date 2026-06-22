#!/bin/bash
# Rerun PopQA Gemini ToG with the exact same ToG method configuration used for
# the DeepSeek and GPT-4o-mini ToG baselines.
#
# This intentionally uses:
#   python3 main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
#
# Outputs:
# - output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/

set -euo pipefail

cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
mkdir -p logs

PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"
MODEL="gemini-2.5-flash-lite"
DATASET="datasets/Popqa"
METHOD_OPT="Option/Method/ToG.yaml"
LIMIT="${LIMIT:-200}"
LOG_FILE="logs/rerun_popqa_tog_gemini_matched_config.log"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/root/.cache/llama_index}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [ "${ALLOW_PARALLEL:-0}" != "1" ]; then
  active_jobs="$(ps -ef | grep -E 'run_20_main|run_popqa_reruns|run_ablation|run_leave_one_out|main.py|newg_main.py' | grep -v grep || true)"
  if [ -n "$active_jobs" ]; then
    echo "Another experiment appears to be running:"
    echo "$active_jobs"
    echo "Exit without starting Gemini ToG matched-config rerun."
    echo "Re-run with ALLOW_PARALLEL=1 only if resource/API contention is intentional."
    exit 1
  fi
fi

set_model() {
  "$PYTHON" - "$1" <<'PY'
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

archive_existing_output() {
  local target_dir="output/datasets/Popqa/ToG_${MODEL}"
  if [ ! -e "$target_dir" ]; then
    echo "no existing Gemini ToG output to archive"
    return 0
  fi

  local ts archive_dir
  ts="$(date +%Y%m%d_%H%M%S)"
  archive_dir=".tog_gemini_matched_config_archived_${ts}"
  mkdir -p "$archive_dir/$(dirname "$target_dir")"
  mv "$target_dir" "$archive_dir/$target_dir"
  echo "archived_old_tog=$archive_dir/$target_dir"
}

summarize_score_file() {
  local label="$1"
  local path="$2"
  "$PYTHON" - "$label" "$path" <<'PY'
import json
import os
import sys

label, path = sys.argv[1], sys.argv[2]
metrics = ["accuracy", "em", "precision", "recall", "f1"]
if not os.path.exists(path):
    print(f"{label}\tMISSING\t{path}")
    raise SystemExit(0)

raw = open(path, encoding="utf-8").read().strip()
try:
    parsed = json.loads(raw)
except json.JSONDecodeError:
    parsed = None

if isinstance(parsed, list):
    rows = [row for row in parsed if isinstance(row, dict)]
elif isinstance(parsed, dict):
    rows = [parsed]
else:
    rows = []
    for line in raw.splitlines():
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
    + f"\t{path}"
)
PY
}

echo "=== Gemini ToG matched-config rerun ==="
echo "model=$MODEL"
echo "method_opt=$METHOD_OPT"
echo "dataset=$DATASET"
echo "limit=$LIMIT"
echo "python=$PYTHON"
echo "log_file=$LOG_FILE"
echo "sentence_transformers_home=$SENTENCE_TRANSFORMERS_HOME"
echo "hf_hub_offline=$HF_HUB_OFFLINE"
echo "transformers_offline=$TRANSFORMERS_OFFLINE"

set_model "$MODEL"
archive_existing_output

echo "=== $(date '+%F %T') run PopQA Gemini ToG matched config ==="
"$PYTHON" main.py -opt "$METHOD_OPT" -dataset_name "$DATASET" --eval_limit "$LIMIT" > "$LOG_FILE" 2>&1

summarize_score_file \
  "PopQA gemini ToG matched-config" \
  "output/datasets/Popqa/ToG_${MODEL}/Results/results.score.json"

echo "=== Gemini ToG matched-config rerun finished $(date '+%F %T') ==="
