# Positive Showcase Figures

This folder contains paper-facing tables and figures extracted from the complete experiment set. Most assets emphasize positive evidence; `showcase_positive_paired_gain_heatmap.png` and `paired_newg_deltas.tsv` now include the full paired-retriever delta grid, including non-positive cells.

## Positive Main Experiment Cases

| Dataset | Model | Strongest baseline | Best NewG | Delta Acc | Delta EM | Delta F1 |
| --- | --- | --- | --- | --- | --- | --- |
| PopQA | DeepSeek | BM25 (28.86) | NewG hippo+bm25 (29.78) | +0.00 | -0.50 | +0.92 |
| PopQA | GPT-4o-mini | BM25 (28.86) | NewG hippo+bm25 (29.06) | -1.00 | -0.50 | +0.20 |
| MuSiQue | DeepSeek | ToG (30.06) | NewG tog+bm25 (31.44) | +0.50 | +0.50 | +1.38 |
| MuSiQue | GPT-4o-mini | HippoRAG (24.90) | NewG hippo+bm25 (31.77) | +2.50 | +5.00 | +6.87 |
| MuSiQue | Gemini | RAPTOR (20.99) | NewG tog+bm25 (30.26) | +10.00 | +7.50 | +9.27 |

## Positive Paired-Baseline Summary

| Dataset | Model | Positive wins | Mean positive F1 gain | Largest F1 gain |
| --- | --- | --- | --- | --- |
| PopQA | DeepSeek | 4/6 | +5.11 | +9.63 |
| PopQA | GPT-4o-mini | 3/6 | +5.05 | +8.17 |
| PopQA | Gemini | 4/6 | +10.34 | +14.00 |
| MuSiQue | DeepSeek | 4/6 | +6.66 | +12.73 |
| MuSiQue | GPT-4o-mini | 5/6 | +6.12 | +7.58 |
| MuSiQue | Gemini | 5/6 | +11.51 | +22.55 |

## Largest Positive LOO Contributions

| Dataset | Component | Metric | Contribution |
| --- | --- | --- | --- |
| PopQA | Answer normalizer | EM | +38.50 |
| PopQA | Answer normalizer | Precision | +21.46 |
| MuSiQue | Answer normalizer | EM | +6.50 |
| MuSiQue | Critic | Recall | +5.50 |
| MuSiQue | Critic | Precision | +5.37 |
| MuSiQue | Critic | F1 | +5.33 |
| MuSiQue | Answer normalizer | Precision | +5.17 |
| MuSiQue | Critic | Accuracy | +4.50 |
| MuSiQue | Critic | EM | +4.00 |
| MuSiQue | Answer normalizer | F1 | +3.59 |
| PopQA | Critic | Accuracy | +2.50 |
| PopQA | Critic | Precision | +2.25 |
| PopQA | Critic | EM | +2.00 |
| PopQA | Answer normalizer | F1 | +1.45 |

## Generated Figures

| Figure | Use |
| --- | --- |
| panel_musique_best_newg_metric_gains.png | Shows Accuracy, EM, and F1 gains of the best NewG method over the strongest baseline on MuSiQue for each model. |
| panel_musique_best_f1_lift.png | Slope chart showing how much F1 rises from the strongest baseline to the best NewG method on MuSiQue. |
| panel_largest_positive_paired_retriever_gains.png | Ranks the largest positive F1 gains when each NewG variant is compared with its paired retriever baseline. |
| panel_leave_one_out_removal_hurts_f1.png | Shows the useful leave-one-out components: removing Critic or Answer normalizer reduces F1. |
| showcase_positive_main_experiment_cards.png | Detailed positive best-NewG cases by metric. |
| showcase_positive_paired_gain_heatmap.png | Complete paired baseline heatmap, including positive and non-positive F1 deltas. |
| showcase_musique_leaderboard.png | MuSiQue paired baseline leaderboard: each model shows HippoRAG, ToG, and RAPTOR with their paired NewG+BM25 and NewG+VDB variants. |
| showcase_musique_bm25_vdb_multimetric_radar.png | Nine-panel MuSiQue radar: for each model and retriever baseline, compares the original baseline with its paired NewG+BM25 and NewG+VDB variants. |
| showcase_popqa_bm25_vdb_multimetric_radar.png | Nine-panel PopQA radar: for each model and retriever baseline, compares the original baseline with its paired NewG+BM25 and NewG+VDB variants. |

## Recommended Main-Paper Set

1. `panel_musique_best_newg_metric_gains.png`
2. `panel_musique_best_f1_lift.png`
3. `showcase_musique_leaderboard.png`
4. `showcase_positive_paired_gain_heatmap.png`
5. `panel_leave_one_out_removal_hurts_f1.png`
6. `showcase_musique_bm25_vdb_multimetric_radar.png`

Use the full mixed/negative results from the parent `figures/` directory in appendix or limitations.
