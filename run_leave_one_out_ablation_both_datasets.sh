#!/bin/bash
# Run the new Full-minus-one NewG ablation plan on both PopQA and MuSiQue.
#
# This script intentionally lists the full variant set, but delegates skipping
# to run_leave_one_out_ablation.sh. Already completed equivalent old results are
# not rerun:
#   full          -> existing main NewG result
#   no_commendor  -> existing NewG_abl_no_commendor result
#   no_normalizer -> existing NewG_abl_critic result
#   single_agent  -> existing NewG_abl_single_agent result
#
# The truly new leave-one-out variants are:
#   no_router, no_regen, no_critic, no_disambiguation
#
# Usage after current MuSiQue ablation is idle:
#   screen -S loo_both
#   bash run_leave_one_out_ablation_both_datasets.sh

set -euo pipefail

cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master

PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"
MODEL="${MODEL:-gemini-2.5-flash-lite}"
GRAPH="${GRAPH:-hipporag}"
TEXT="${TEXT:-bm25}"
LIMIT="${LIMIT:-200}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
VARIANTS="${VARIANTS:-full no_router no_regen no_critic no_commendor no_normalizer no_disambiguation single_agent}"
DATASETS="${DATASETS:-datasets/Popqa datasets/musique}"

export PYTHON MODEL GRAPH TEXT LIMIT SKIP_COMPLETED VARIANTS
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/root/.cache/llama_index}"

echo "=== NewG leave-one-out ablation on both datasets ==="
echo "model=$MODEL"
echo "graph=$GRAPH"
echo "text=$TEXT"
echo "limit=$LIMIT"
echo "variants=$VARIANTS"
echo "datasets=$DATASETS"
echo "skip_completed=$SKIP_COMPLETED"

for dataset in $DATASETS; do
  echo
  echo "============================================================"
  echo "DATASET=$dataset"
  echo "============================================================"

  DATA="$dataset" bash run_leave_one_out_ablation.sh

  "$PYTHON" collect_leave_one_out_ablation.py \
    --dataset "$dataset" \
    --graph "$GRAPH" \
    --text "$TEXT" \
    --model "$MODEL" \
    --variants $VARIANTS

  "$PYTHON" plot_leave_one_out_ablation.py \
    --dataset "$dataset" \
    --graph "$GRAPH" \
    --text "$TEXT" \
    --model "$MODEL" \
    --variants $VARIANTS
done

echo "=== Both-dataset leave-one-out ablation workflow finished ==="
