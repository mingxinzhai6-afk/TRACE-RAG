#!/bin/bash
# NewG experiment serial runner (gemini-2.5-flash-lite)
# This is not the baseline-inclusive 20-run suite; use run_20_main.sh for that.
#
#  1-7   PopQA Ablation
#  8-14  MuSiQue Ablation
# 15-20  PopQA NewG Main (3 graph x 2 text)
# 21-26  MuSiQue NewG Main (3 graph x 2 text)

set -e

cd "${HOME}/autodl-tmp/GraphRAG-master/GraphRAG-master"
mkdir -p logs

# ===== 1-7. PopQA Ablation =====
echo "=== [1-7] PopQA Ablation ==="
bash run_ablation_popqa.sh > logs/all20_popqa_abl.log 2>&1

# ===== 8-14. MuSiQue Ablation =====
echo "=== [8-14] MuSiQue Ablation ==="
bash run_ablation_musique.sh > logs/all20_musique_abl.log 2>&1

# ===== 15-20. PopQA Main (6 combos) =====
echo "=== [15-20] PopQA Main ==="
for graph in hipporag tog raptor; do
  for text in bm25 vdb; do
    echo "=== PopQA: ${graph}+${text} ==="
    python newg_main.py \
      -opt Option/Method/NewG.yaml \
      -graph_method ${graph} \
      -text_method ${text} \
      -dataset_name datasets/Popqa \
      --eval_limit 200 > logs/all20_popqa_${graph}_${text}.log 2>&1
  done
done

# ===== 21-26. MuSiQue Main (6 combos) =====
echo "=== [21-26] MuSiQue Main ==="
for graph in hipporag tog raptor; do
  for text in bm25 vdb; do
    echo "=== MuSiQue: ${graph}+${text} ==="
    python newg_main.py \
      -opt Option/Method/NewG.yaml \
      -graph_method ${graph} \
      -text_method ${text} \
      -dataset_name datasets/musique \
      --eval_limit 200 > logs/all20_musique_${graph}_${text}.log 2>&1
  done
done

echo "=== All 26 NewG experiments finished ==="
