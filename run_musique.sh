#!/bin/bash
# MuSiQue full pipeline: data prep → build graphs → baselines → NewG 6 combos
# Usage:
#   nohup bash run_musique.sh > run_musique.log 2>&1 &
#   tail -f run_musique.log   # monitor progress

set -e  # stop on first error
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── 1. Build graphs ─────────────────────────────────────────────────────────
log "=== STEP 1a: Build er_graph_colbert (HippoRAG) ==="
python build_graph.py \
  -opt Option/Method/HippoRAG.yaml \
  -dataset_name datasets/musique
log "=== STEP 1a done ==="

log "=== STEP 1b: Build er_graph (ToG) ==="
python build_graph.py \
  -opt Option/Method/ToG.yaml \
  -dataset_name datasets/musique
log "=== STEP 1b done ==="

log "=== STEP 1c: Build tree_graph_balanced (RAPTOR) ==="
python build_graph.py \
  -opt Option/Method/RAPTOR.yaml \
  -dataset_name datasets/musique
log "=== STEP 1c done ==="

# ── 2. Baselines ─────────────────────────────────────────────────────────────
log "=== STEP 2a: HippoRAG baseline ==="
python main.py \
  -opt Option/Method/HippoRAG.yaml \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 2a done ==="

log "=== STEP 2b: RAPTOR baseline ==="
python main.py \
  -opt Option/Method/RAPTOR.yaml \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 2b done ==="

log "=== STEP 2c: ToG baseline ==="
python main.py \
  -opt Option/Method/ToG.yaml \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 2c done ==="

log "=== STEP 2d: AgentG baseline ==="
python main.py \
  -opt Option/Method/AgentG.yaml \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 2d done ==="

# ── 3. NewG 6 combos ─────────────────────────────────────────────────────────
log "=== STEP 3a: NewG hipporag+bm25 ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method hipporag -text_method bm25 \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 3a done ==="

log "=== STEP 3b: NewG hipporag+vdb ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method hipporag -text_method vdb \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 3b done ==="

log "=== STEP 3c: NewG tog+bm25 ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method tog -text_method bm25 \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 3c done ==="

log "=== STEP 3d: NewG tog+vdb ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method tog -text_method vdb \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 3d done ==="

log "=== STEP 3e: NewG raptor+bm25 ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method raptor -text_method bm25 \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 3e done ==="

log "=== STEP 3f: NewG raptor+vdb ==="
python newg_main.py \
  -opt Option/Method/NewG.yaml \
  -graph_method raptor -text_method vdb \
  -dataset_name datasets/musique --eval_limit 200
log "=== STEP 3f done ==="

log "=== ALL DONE: MuSiQue pipeline complete ==="
