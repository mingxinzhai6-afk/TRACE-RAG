#!/bin/bash
# Ablation experiments on PopQA (hipporag+bm25, 200 queries).
#
# Main ladder:
#   Step 1: Simple-Gen       fixed graph, direct EvidenceFusion answer
#   Step 2: +Routing         add QueryUnderstanding adaptive routing
#   Step 3: +ReGeneration    add 3-judge / 3-voter ReGenerationAgent
#   Step 4: +Critic          add Critic-driven iterative retrieval + Commendor
#   Full NewG is reused from the main result:
#     output/datasets/Popqa/NewG_hipporag_bm25_<model>/
#
# Horizontal controls:
#   Step 5: Full w/o Commendor
#   Step 6: Full 1 judge / 1 voter
#
# Usage: screen -S abl && bash run_ablation_popqa.sh

set -e

MODEL=gemini-2.5-flash-lite
GRAPH=hipporag
TEXT=bm25
DATA=datasets/Popqa
LIMIT=200
PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"

echo "=== Switch llm.model to ${MODEL} for PopQA ablation ==="
"${PYTHON}" - "${MODEL}" <<'PY'
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

echo "=== [1/7] Simple-Gen ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_simple.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT}

echo "=== [2/7] +Routing ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_routing.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT}

echo "=== [3/7] +ReGeneration ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_regen.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT}

echo "=== [4/7] +Critic ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_critic.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT}

echo "=== [5/6] Full w/o Commendor ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_no_commendor.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT}

echo "=== [6/6] Full 1 judge / 1 voter ==="
"${PYTHON}" newg_main.py \
  -opt Option/Method/NewG_abl_single_agent.yaml \
  -graph_method ${GRAPH} \
  -text_method ${TEXT} \
  -dataset_name ${DATA} \
  --eval_limit ${LIMIT}

echo "=== PopQA ablation done. Results saved under output/datasets/Popqa/ ==="
ls -d output/datasets/Popqa/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_*
