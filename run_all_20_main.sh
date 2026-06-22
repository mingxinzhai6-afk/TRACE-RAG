#!/bin/bash
# 20 main experiments serial runner (gemini-2.5-flash-lite)
#  PopQA:  HippoRAG, RAPTOR, ToG, AgentG, NewG x6
#  MuSiQue: HippoRAG, RAPTOR, ToG, AgentG, NewG x6

set -e
cd "${HOME}/autodl-tmp/GraphRAG-master/GraphRAG-master"
mkdir -p logs

# ===== PopQA (10) =====
python main.py -opt Option/Method/HippoRAG.yaml -dataset_name datasets/Popqa --eval_limit 200 > logs/popqa_hipporag.log 2>&1
python main.py -opt Option/Method/RAPTOR.yaml   -dataset_name datasets/Popqa --eval_limit 200 > logs/popqa_raptor.log  2>&1
python main.py -opt Option/Method/ToG.yaml      -dataset_name datasets/Popqa --eval_limit 200 > logs/popqa_tog.log     2>&1
python main.py -opt Option/Method/AgentG.yaml   -dataset_name datasets/Popqa --eval_limit 200 > logs/popqa_agentg.log  2>&1

for g in hipporag tog raptor; do
  for t in bm25 vdb; do
    python newg_main.py -opt Option/Method/NewG.yaml -graph_method $g -text_method $t -dataset_name datasets/Popqa --eval_limit 200 > logs/popqa_newg_${g}_${t}.log 2>&1
  done
done

# ===== MuSiQue (10) =====
python main.py -opt Option/Method/HippoRAG.yaml -dataset_name datasets/musique --eval_limit 200 > logs/musique_hipporag.log 2>&1
python main.py -opt Option/Method/RAPTOR.yaml   -dataset_name datasets/musique --eval_limit 200 > logs/musique_raptor.log  2>&1
python main.py -opt Option/Method/ToG.yaml      -dataset_name datasets/musique --eval_limit 200 > logs/musique_tog.log     2>&1
python main.py -opt Option/Method/AgentG.yaml   -dataset_name datasets/musique --eval_limit 200 > logs/musique_agentg.log  2>&1

for g in hipporag tog raptor; do
  for t in bm25 vdb; do
    python newg_main.py -opt Option/Method/NewG.yaml -graph_method $g -text_method $t -dataset_name datasets/musique --eval_limit 200 > logs/musique_newg_${g}_${t}.log 2>&1
  done
done

echo "=== ALL 20 MAIN EXPERIMENTS DONE ==="
