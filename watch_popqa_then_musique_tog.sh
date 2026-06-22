#!/bin/bash
# Wait for PopQA ablation to finish, then run MuSiQue ablation and the normal
# Gemini ToG rerun from one orchestrator.
#
# Default order is sequential to avoid API/GPU contention:
#   1. bash run_ablation_musique.sh
#   2. bash run_tog_gemini_v3.sh
#
# If you intentionally want true parallel execution after PopQA finishes:
#   RUN_PARALLEL=1 bash watch_popqa_then_musique_tog.sh

set -euo pipefail

cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
mkdir -p logs

POLL_SECONDS="${POLL_SECONDS:-30}"
RUN_PARALLEL="${RUN_PARALLEL:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/watch_popqa_then_musique_tog_${STAMP}.log"
LOCK_DIR=".watch_popqa_then_musique_tog.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another watcher appears to be running: $LOCK_DIR"
  echo "If this is stale, remove it manually after confirming no watcher is active."
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

exec > >(tee -a "$LOG_FILE") 2>&1

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

has_popqa_ablation() {
  pgrep -af 'bash[[:space:]]+run_ablation_popqa\.sh|run_ablation_popqa\.sh' \
    | grep -v 'watch_popqa_then_musique_tog' >/dev/null 2>&1
}

has_popqa_newg_child() {
  pgrep -af 'newg_main\.py.*NewG_abl_.*datasets/Popqa|newg_main\.py.*datasets/Popqa.*NewG_abl_' \
    >/dev/null 2>&1
}

echo "=== watcher started $(date '+%F %T %z') ==="
echo "log_file=$LOG_FILE"
echo "poll_seconds=$POLL_SECONDS"
echo "run_parallel=$RUN_PARALLEL"

while has_popqa_ablation || has_popqa_newg_child; do
  echo "--- $(date '+%F %T') PopQA ablation still active ---"
  ps -ef | grep -E 'run_ablation_popqa|newg_main.py.*datasets/Popqa' | grep -v grep || true
  sleep "$POLL_SECONDS"
done

echo "=== $(date '+%F %T') PopQA ablation is no longer active ==="
set_model gemini-2.5-flash-lite

if [ "$RUN_PARALLEL" = "1" ]; then
  echo "=== $(date '+%F %T') starting MuSiQue ablation and Gemini ToG rerun in parallel ==="
  bash run_ablation_musique.sh > "logs/auto_ablation_musique_${STAMP}.log" 2>&1 &
  musique_pid="$!"
  ALLOW_PARALLEL=1 bash run_tog_gemini_v3.sh > "logs/auto_tog_gemini_${STAMP}.log" 2>&1 &
  tog_pid="$!"

  echo "musique_pid=$musique_pid log=logs/auto_ablation_musique_${STAMP}.log"
  echo "tog_pid=$tog_pid log=logs/auto_tog_gemini_${STAMP}.log"

  musique_status=0
  tog_status=0
  wait "$musique_pid" || musique_status="$?"
  wait "$tog_pid" || tog_status="$?"
  echo "musique_status=$musique_status"
  echo "tog_status=$tog_status"
  if [ "$musique_status" != "0" ] || [ "$tog_status" != "0" ]; then
    exit 1
  fi
else
  echo "=== $(date '+%F %T') starting MuSiQue ablation ==="
  bash run_ablation_musique.sh
  echo "=== $(date '+%F %T') MuSiQue ablation finished; starting Gemini ToG rerun ==="
  bash run_tog_gemini_v3.sh
fi

echo "=== watcher workflow finished $(date '+%F %T %z') ==="
