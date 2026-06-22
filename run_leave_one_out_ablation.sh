#!/bin/bash
# Full-minus-one NewG ablation runner.
#
# This runner keeps NewG.yaml as the single full-system source of truth, then
# generates one config per ablation variant:
#   full, no_router, no_regen, no_critic, no_commendor,
#   no_normalizer, no_disambiguation, single_agent
#
# Defaults target the PopQA setup that exposed the current ablation issue:
#   graph=hipporag, text=bm25, model=gemini-2.5-flash-lite, n=200
#
# Example:
#   screen -S loo_popqa
#   bash run_leave_one_out_ablation.sh
#
# Override examples:
#   DATA=datasets/musique bash run_leave_one_out_ablation.sh
#   VARIANTS="full no_router no_regen no_normalizer" bash run_leave_one_out_ablation.sh

set -euo pipefail

cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master

MODEL="${MODEL:-gemini-2.5-flash-lite}"
GRAPH="${GRAPH:-hipporag}"
TEXT="${TEXT:-bm25}"
DATA="${DATA:-datasets/Popqa}"
LIMIT="${LIMIT:-200}"
PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"
CONFIG_DIR="${CONFIG_DIR:-Option/Method/generated_leave_one_out}"
VARIANTS="${VARIANTS:-full no_router no_regen no_critic no_commendor no_normalizer no_disambiguation single_agent}"
ALLOW_PARALLEL="${ALLOW_PARALLEL:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export PYTHON
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/root/.cache/llama_index}"

mkdir -p logs "$CONFIG_DIR"

dataset_slug="$(basename "$DATA")"
model_slug="$(printf '%s' "$MODEL" | tr '/:' '__')"
model_short="$("$PYTHON" - "$MODEL" <<'PY'
import re
import sys

model = sys.argv[1]
short = model.split("/")[-1].split(":")[0]
short = re.sub(r"(?i)-instruct.*", "", short)
print(re.sub(r"[^\w\-.]", "_", short).strip("_"))
PY
)"

if [ "$ALLOW_PARALLEL" != "1" ]; then
  active_jobs="$(
    ps -ef \
      | grep -E 'run_20_main|run_popqa_reruns|run_ablation|run_leave_one_out_ablation|main.py|newg_main.py' \
      | grep -v grep \
      | grep -v "bash run_leave_one_out_ablation.sh" \
      | grep -v "bash run_leave_one_out_ablation_both_datasets.sh" || true
  )"
  if [ -n "$active_jobs" ]; then
    echo "Another experiment appears to be running:"
    echo "$active_jobs"
    echo "Exit without starting leave-one-out ablation."
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
    r'(?m)^(\s*model:\s*)"[^"]+"',
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

score_candidates() {
  local results_dir="$1"
  printf '%s\n' \
    "$results_dir/results.score.json" \
    "$results_dir/results.score.jsonl" \
    "$results_dir/results.json" \
    "$results_dir/results.jsonl"
}

variant_result_dirs() {
  local variant="$1"
  local prefix="output/${DATA}/NewG_${GRAPH}_${TEXT}_${model_short}"

  case "$variant" in
    full)
      printf '%s\n' \
        "${prefix}_loo_full/Results" \
        "${prefix}/Results" \
        "${prefix}_abl_normalizer/Results"
      ;;
    no_commendor)
      printf '%s\n' \
        "${prefix}_loo_no_commendor/Results" \
        "${prefix}_abl_no_commendor/Results"
      ;;
    no_normalizer)
      printf '%s\n' \
        "${prefix}_loo_no_normalizer/Results" \
        "${prefix}_abl_critic/Results"
      ;;
    single_agent)
      printf '%s\n' \
        "${prefix}_loo_single_agent/Results" \
        "${prefix}_abl_single_agent/Results"
      ;;
    *)
      printf '%s\n' "${prefix}_loo_${variant}/Results"
      ;;
  esac
}

score_row_count() {
  local score_file="$1"
  "$PYTHON" - "$score_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        print(0)
        raise SystemExit
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        print(len([row for row in parsed if isinstance(row, dict)]))
    elif isinstance(parsed, dict):
        print(1)
    else:
        count = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            json.loads(line)
            count += 1
        print(count)
except Exception:
    print("ERR")
PY
}

find_existing_score_file() {
  local variant="$1"
  local require_complete="${2:-0}"
  local result_dir candidate n

  while IFS= read -r result_dir; do
    while IFS= read -r candidate; do
      if [ ! -f "$candidate" ]; then
        continue
      fi
      if [ "$require_complete" = "1" ]; then
        n="$(score_row_count "$candidate")"
        if [ "$n" != "$LIMIT" ]; then
          continue
        fi
      fi
      printf '%s\n' "$candidate"
      return 0
    done < <(score_candidates "$result_dir")
  done < <(variant_result_dirs "$variant")

  return 1
}

summarize_score_file() {
  local variant="$1"
  local score_file="$2"

  "$PYTHON" - "$variant" "$score_file" <<'PY'
import json
import sys
from pathlib import Path

variant, path = sys.argv[1], sys.argv[2]
metrics = ["accuracy", "em", "precision", "recall", "f1"]
if not path:
    print(f"{variant}\tMISSING")
    raise SystemExit(0)

raw = Path(path).read_text(encoding="utf-8").strip()
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
    f"{variant}\tn={len(rows)}\t"
    + "\t".join(f"{metric}={value * 100:.2f}%" for metric, value in zip(metrics, vals))
    + f"\t{path}"
)
PY
}

summarize_variant() {
  local variant="$1"
  local score_file
  score_file="$(find_existing_score_file "$variant" 0 || true)"
  if [ -z "$score_file" ]; then
    echo "${variant}	MISSING"
    return 0
  fi
  summarize_score_file "$variant" "$score_file"
}

echo "=== Leave-one-out NewG ablation ==="
echo "model=$MODEL"
echo "graph=$GRAPH"
echo "text=$TEXT"
echo "data=$DATA"
echo "limit=$LIMIT"
echo "python=$PYTHON"
echo "sentence_transformers_home=$SENTENCE_TRANSFORMERS_HOME"
echo "variants=$VARIANTS"
echo "skip_completed=$SKIP_COMPLETED"

echo "=== Switch llm.model to ${MODEL} ==="
set_model "$MODEL"

echo "=== Generate leave-one-out configs ==="
"$PYTHON" generate_newg_leave_one_out_configs.py --out-dir "$CONFIG_DIR" --variants $VARIANTS

for variant in $VARIANTS; do
  opt="${CONFIG_DIR}/NewG_loo_${variant}.yaml"
  log="logs/loo_${dataset_slug}_${GRAPH}_${TEXT}_${model_slug}_${variant}.log"
  if [ ! -f "$opt" ]; then
    echo "Missing generated config: $opt"
    exit 1
  fi

  if [ "$SKIP_COMPLETED" = "1" ]; then
    existing_complete="$(find_existing_score_file "$variant" 1 || true)"
    if [ -n "$existing_complete" ]; then
      echo "=== $(date '+%F %T') skip variant=${variant}; complete equivalent result exists ==="
      summarize_score_file "$variant" "$existing_complete"
      continue
    fi
  fi

  echo "=== $(date '+%F %T') variant=${variant} opt=${opt} ==="
  "$PYTHON" newg_main.py \
    -opt "$opt" \
    -graph_method "$GRAPH" \
    -text_method "$TEXT" \
    -dataset_name "$DATA" \
    --eval_limit "$LIMIT" > "$log" 2>&1

  echo "--- summary ${variant} ---"
  summarize_variant "$variant"
done

echo "=== Leave-one-out ablation done ==="
