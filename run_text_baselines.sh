#!/bin/bash
# Text-only baselines for comparison tables.
# Runs BM25 and Entity-VDB baselines on PopQA and MuSiQue.

set -e

cd "${HOME}/autodl-tmp/GraphRAG-master/GraphRAG-master"
mkdir -p logs

echo "=== PopQA: BM25 ==="
python main.py -opt Option/Method/BM25.yaml -dataset_name datasets/Popqa --eval_limit 200 > logs/popqa_bm25.log 2>&1

echo "=== PopQA: VDB ==="
python main.py -opt Option/Method/VDB.yaml -dataset_name datasets/Popqa --eval_limit 200 > logs/popqa_vdb.log 2>&1

echo "=== MuSiQue: BM25 ==="
python main.py -opt Option/Method/BM25.yaml -dataset_name datasets/musique --eval_limit 200 > logs/musique_bm25.log 2>&1

echo "=== MuSiQue: VDB ==="
python main.py -opt Option/Method/VDB.yaml -dataset_name datasets/musique --eval_limit 200 > logs/musique_vdb.log 2>&1

echo "=== Text baselines done ==="
