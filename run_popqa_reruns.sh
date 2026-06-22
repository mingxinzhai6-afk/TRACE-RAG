#!/bin/bash
# PopQA + MuSiQue reruns for missing/suspicious cross-model baseline cells.
#
# Run on the server:
#   cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
#   bash run_popqa_reruns.sh
#
# Set ALLOW_PARALLEL=1 only if you intentionally want to run while another
# main.py/newg_main.py job is active.

set -euo pipefail

cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
mkdir -p logs

if [ "${ALLOW_PARALLEL:-0}" != "1" ]; then
  active_jobs="$(ps -ef | grep -E 'run_20_main|main.py|newg_main.py' | grep -v grep || true)"
  if [ -n "$active_jobs" ]; then
    echo "Another experiment appears to be running:"
    echo "$active_jobs"
    echo "Exit without starting reruns. Re-run with ALLOW_PARALLEL=1 to override."
    exit 1
  fi
fi

backup_outputs() {
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local targets=(
    output/datasets/Popqa/ToG_gemini-2.5-flash-lite
    output/datasets/Popqa/BM25_deepseek-v3.2
    output/datasets/Popqa/VDB_deepseek-v3.2
    output/datasets/Popqa/HippoRAG_deepseek-v3.2
    output/datasets/Popqa/BM25_gpt-4o-mini
    output/datasets/Popqa/VDB_gpt-4o-mini
    output/datasets/musique/BM25_deepseek-v3.2
    output/datasets/musique/VDB_deepseek-v3.2
    output/datasets/musique/BM25_gpt-4o-mini
    output/datasets/musique/VDB_gpt-4o-mini
  )
  local existing=()
  local d
  for d in "${targets[@]}"; do
    [ -e "$d" ] && existing+=("$d")
  done
  if [ "${#existing[@]}" -gt 0 ]; then
    tar -czf ".popqa_rerun_backup_${ts}.tar.gz" "${existing[@]}"
    echo "backup=.popqa_rerun_backup_${ts}.tar.gz"
    local archive_dir=".popqa_rerun_archived_${ts}"
    mkdir -p "$archive_dir"
    for d in "${existing[@]}"; do
      mkdir -p "$archive_dir/$(dirname "$d")"
      mv "$d" "$archive_dir/$d"
    done
    echo "archived_dirs=$archive_dir"
  else
    echo "no existing target outputs to back up"
  fi
}

set_model() {
  python3 - "$1" <<'PY'
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

run_one() {
  local model="$1"
  local method="$2"
  local dataset="$3"
  local log_name="$4"
  set_model "$model"
  echo "=== $(date '+%F %T') ${model} ${dataset} ${method} ==="
  python3 main.py -opt "Option/Method/${method}.yaml" -dataset_name "datasets/${dataset}" --eval_limit 200 \
    > "logs/${log_name}" 2>&1
}

backup_outputs

run_one gemini-2.5-flash-lite ToG Popqa rerun_popqa_tog_gemini.log

run_one deepseek-v3.2 BM25 Popqa rerun_popqa_bm25_deepseek.log
run_one deepseek-v3.2 VDB Popqa rerun_popqa_vdb_deepseek.log
run_one deepseek-v3.2 HippoRAG Popqa rerun_popqa_hipporag_deepseek.log

run_one gpt-4o-mini BM25 Popqa rerun_popqa_bm25_gpt4omini.log
run_one gpt-4o-mini VDB Popqa rerun_popqa_vdb_gpt4omini.log

run_one deepseek-v3.2 BM25 musique rerun_musique_bm25_deepseek.log
run_one deepseek-v3.2 VDB musique rerun_musique_vdb_deepseek.log

run_one gpt-4o-mini BM25 musique rerun_musique_bm25_gpt4omini.log
run_one gpt-4o-mini VDB musique rerun_musique_vdb_gpt4omini.log

python3 - <<'PY'
import json
import os

paths = [
    ("PopQA", "gemini", "ToG", "output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.score.json"),
    ("PopQA", "deepseek", "BM25", "output/datasets/Popqa/BM25_deepseek-v3.2/Results/results.score.json"),
    ("PopQA", "deepseek", "VDB", "output/datasets/Popqa/VDB_deepseek-v3.2/Results/results.score.json"),
    ("PopQA", "deepseek", "HippoRAG", "output/datasets/Popqa/HippoRAG_deepseek-v3.2/Results/results.score.json"),
    ("PopQA", "gpt-4o-mini", "BM25", "output/datasets/Popqa/BM25_gpt-4o-mini/Results/results.score.json"),
    ("PopQA", "gpt-4o-mini", "VDB", "output/datasets/Popqa/VDB_gpt-4o-mini/Results/results.score.json"),
    ("MuSiQue", "deepseek", "BM25", "output/datasets/musique/BM25_deepseek-v3.2/Results/results.score.json"),
    ("MuSiQue", "deepseek", "VDB", "output/datasets/musique/VDB_deepseek-v3.2/Results/results.score.json"),
    ("MuSiQue", "gpt-4o-mini", "BM25", "output/datasets/musique/BM25_gpt-4o-mini/Results/results.score.json"),
    ("MuSiQue", "gpt-4o-mini", "VDB", "output/datasets/musique/VDB_gpt-4o-mini/Results/results.score.json"),
]
metrics = ["accuracy", "em", "precision", "recall", "f1"]

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

print("dataset\tmodel\tmethod\tn\taccuracy\tem\tprecision\trecall\tf1")
for dataset, model, method, path in paths:
    if not os.path.exists(path):
        print(f"{dataset}\t{model}\t{method}\tMISSING")
        continue
    rows = load_rows(path)
    vals = [
        sum(float(r.get(metric, 0) or 0) for r in rows) / len(rows)
        for metric in metrics
    ]
    print(
        f"{dataset}\t{model}\t{method}\t{len(rows)}\t"
        + "\t".join(f"{v * 100:.2f}%" for v in vals)
    )
PY

echo "=== PopQA + MuSiQue reruns finished ==="
