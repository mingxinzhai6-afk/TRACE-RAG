#!/bin/bash
# PopQA: NewG 6 combos (graphs already built) + ToG/AgentG baselines
# Usage:
#   nohup bash run_popqa_newg.sh > run_popqa_newg.log 2>&1 &
#   tail -f run_popqa_newg.log

set -e
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== STEP 1a: NewG hipporag+bm25 ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method hipporag -text_method bm25 \
  -dataset_name datasets/Popqa --eval_limit 200
log "=== STEP 1a done ==="

log "=== STEP 1b: NewG hipporag+vdb ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method hipporag -text_method vdb \
  -dataset_name datasets/Popqa --eval_limit 200
log "=== STEP 1b done ==="

log "=== STEP 1c: NewG tog+bm25 ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method tog -text_method bm25 \
  -dataset_name datasets/Popqa --eval_limit 200
log "=== STEP 1c done ==="

log "=== STEP 1d: NewG tog+vdb ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method tog -text_method vdb \
  -dataset_name datasets/Popqa --eval_limit 200
log "=== STEP 1d done ==="

log "=== STEP 1e: NewG raptor+bm25 ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method raptor -text_method bm25 \
  -dataset_name datasets/Popqa --eval_limit 200
log "=== STEP 1e done ==="

log "=== STEP 1f: NewG raptor+vdb ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method raptor -text_method vdb \
  -dataset_name datasets/Popqa --eval_limit 200
log "=== STEP 1f done ==="

log "=== STEP 2a: ToG baseline ==="
python main.py \
  -opt Option/Method/ToG.yaml \
  -dataset_name datasets/Popqa --eval_limit 200
log "=== STEP 2a done ==="

log "=== STEP 2b: AgentG baseline ==="
python main.py \
  -opt Option/Method/AgentG.yaml \
  -dataset_name datasets/Popqa --eval_limit 200
log "=== STEP 2b done ==="

log "=== ALL DONE: PopQA pipeline complete ==="
