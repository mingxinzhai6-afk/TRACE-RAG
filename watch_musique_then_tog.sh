#!/bin/bash
# Wait for the currently running MuSiQue ablation to finish, then run the
# normal PopQA Gemini ToG rerun. This is intended for the case where
# run_ablation_musique.sh was started manually or in screen after the broader
# PopQA->MuSiQue->ToG watcher exited.

set -euo pipefail

cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
mkdir -p logs

POLL_SECONDS="${POLL_SECONDS:-60}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/watch_musique_then_tog_${STAMP}.log"
LOCK_DIR=".watch_musique_then_tog.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another MuSiQue->ToG watcher appears to be running: $LOCK_DIR"
  echo "If this is stale, remove it manually after confirming no watcher is active."
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/root/.cache/llama_index}"

has_musique_ablation() {
  pgrep -af 'bash[[:space:]]+run_ablation_musique\.sh|run_ablation_musique\.sh' \
    | grep -v 'watch_musique_then_tog' >/dev/null 2>&1
}

has_musique_newg_child() {
  pgrep -af 'newg_main\.py.*NewG_abl_.*datasets/musique|newg_main\.py.*datasets/musique.*NewG_abl_' \
    >/dev/null 2>&1
}

check_ablation_scores() {
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

dirs = [
    Path("output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_simple/Results"),
    Path("output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_routing/Results"),
    Path("output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_regen/Results"),
    Path("output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_critic/Results"),
    Path("output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_no_commendor/Results"),
    Path("output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_single_agent/Results"),
]

def score_file(results_dir: Path) -> Path | None:
    for name in ("results.score.json", "results.score.jsonl"):
        path = results_dir / name
        if path.exists():
            return path
    return None

ok = True
for results_dir in dirs:
    path = score_file(results_dir)
    if path is None:
        print(f"MISSING\t{results_dir}/results.score.json[l]")
        ok = False
        continue
    if not path.exists():
        print(f"MISSING\t{path}")
        ok = False
        continue
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"COUNT\t{len(rows)}\t{path}")
    if len(rows) != 200:
        ok = False

raise SystemExit(0 if ok else 1)
PY
}

echo "=== MuSiQue->ToG watcher started $(date '+%F %T %z') ==="
echo "log_file=$LOG_FILE"
echo "poll_seconds=$POLL_SECONDS"
echo "python=$PYTHON"
echo "sentence_transformers_home=$SENTENCE_TRANSFORMERS_HOME"

while has_musique_ablation || has_musique_newg_child; do
  echo "--- $(date '+%F %T') MuSiQue ablation still active ---"
  ps -ef | grep -E 'run_ablation_musique|newg_main.py.*datasets/musique' | grep -v grep || true
  sleep "$POLL_SECONDS"
done

echo "=== $(date '+%F %T') MuSiQue ablation is no longer active ==="
echo "=== checking MuSiQue ablation score files ==="
if ! check_ablation_scores; then
  echo "MuSiQue ablation score files are missing or incomplete; not starting ToG."
  exit 1
fi

echo "=== $(date '+%F %T') starting PopQA Gemini ToG rerun ==="
bash run_tog_gemini_v3.sh

echo "=== MuSiQue->ToG watcher workflow finished $(date '+%F %T %z') ==="
