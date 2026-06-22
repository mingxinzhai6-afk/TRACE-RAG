#!/bin/bash
# 24 main experiments (gemini-2.5-flash-lite)
# PopQA:  BM25, VDB, HippoRAG, RAPTOR, ToG, AgentG, NewG x6
# MuSiQue: BM25, VDB, HippoRAG, RAPTOR, ToG, AgentG, NewG x6

set -e
cd "${HOME}/autodl-tmp/GraphRAG-master/GraphRAG-master"

# Use python3 by default on servers where `python` is not installed.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ===== PopQA (12) =====
"${PYTHON_BIN}" main.py -opt Option/Method/BM25.yaml     -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/VDB.yaml      -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/HippoRAG.yaml -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/RAPTOR.yaml   -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/ToG.yaml      -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/AgentG.yaml   -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method hipporag -text_method vdb  -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method tog      -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method tog      -text_method vdb  -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method raptor   -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method raptor   -text_method vdb  -dataset_name datasets/Popqa --eval_limit 200

# ===== MuSiQue (12) =====
"${PYTHON_BIN}" main.py -opt Option/Method/BM25.yaml     -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/VDB.yaml      -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/HippoRAG.yaml -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/RAPTOR.yaml   -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/ToG.yaml      -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" main.py -opt Option/Method/AgentG.yaml   -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method hipporag -text_method vdb  -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method tog      -text_method bm25 -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method tog      -text_method vdb  -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method raptor   -text_method bm25 -dataset_name datasets/musique --eval_limit 200 && \
"${PYTHON_BIN}" newg_main.py -opt Option/Method/NewG.yaml -graph_method raptor   -text_method vdb  -dataset_name datasets/musique --eval_limit 200

echo "=== ALL 24 MAIN EXPERIMENTS DONE ==="
