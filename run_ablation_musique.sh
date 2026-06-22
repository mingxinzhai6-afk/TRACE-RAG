#!/bin/bash
# Ablation experiments on MuSiQue (hipporag+bm25, 200 queries).
#
# Main ladder:
#   Step 1: Simple-Gen       fixed graph, direct EvidenceFusion answer
#   Step 2: +Routing         add QueryUnderstanding adaptive routing
#   Step 3: +ReGeneration    add 3-judge / 3-voter ReGenerationAgent
#   Step 4: +Critic          add Critic-driven iterative retrieval + Commendor
#   Full NewG is reused from the main result:
#     output/datasets/musique/NewG_hipporag_bm25_<model>/
#
# Horizontal controls:
#   Step 5: Full w/o Commendor
#   Step 6: Full 1 judge / 1 voter
#
# Usage:
#   screen -S abl_musique && bash run_ablation_musique.sh
#   nohup bash run_ablation_musique.sh > abl_musique.log 2>&1 &

set -e

GRAPH=hipporag
TEXT=bm25
DATA=datasets/musique
LIMIT=200
PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"

mkdir -p logs

echo "=== [1/7] Simple-Gen ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_simple.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT} > logs/abl_musique_step1_simple.log 2>&1

echo "=== [2/7] +Routing ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_routing.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT} > logs/abl_musique_step2_routing.log 2>&1

echo "=== [3/7] +ReGeneration ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_regen.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT} > logs/abl_musique_step3_regen.log 2>&1

echo "=== [4/7] +Critic ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_critic.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT} > logs/abl_musique_step4_critic.log 2>&1

echo "=== [5/6] Full w/o Commendor ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_no_commendor.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT} > logs/abl_musique_step5_no_commendor.log 2>&1

echo "=== [6/6] Full 1 judge / 1 voter ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_single_agent.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT} > logs/abl_musique_step6_single_agent.log 2>&1

echo "=== MuSiQue ablation done. Results saved under output/datasets/musique/ ==="
ls -d output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_*
