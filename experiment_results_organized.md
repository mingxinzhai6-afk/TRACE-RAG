# Experiment Result Organization

This report keeps the full result structure, while separating results that are strong enough for main-paper narrative from mixed or negative findings better suited for appendix/limitations.

## Main Experiments: Best Baseline vs Best NewG

| Dataset | Model | Best baseline F1 | Best NewG F1 | Delta Acc | Delta EM | Delta F1 |
| --- | --- | --- | --- | --- | --- | --- |
| PopQA | DeepSeek | BM25 (28.86) | NewG hippo+bm25 (29.78) | +0.00 | -0.50 | +0.92 |
| PopQA | GPT-4o-mini | BM25 (28.86) | NewG hippo+bm25 (29.06) | -1.00 | -0.50 | +0.20 |
| PopQA | Gemini | HippoRAG (27.85) | NewG hippo+bm25 (27.50) | -2.00 | +0.00 | -0.35 |
| MuSiQue | DeepSeek | ToG (30.06) | NewG tog+bm25 (31.44) | +0.50 | +0.50 | +1.38 |
| MuSiQue | GPT-4o-mini | HippoRAG (24.90) | NewG hippo+bm25 (31.77) | +2.50 | +5.00 | +6.87 |
| MuSiQue | Gemini | RAPTOR (20.99) | NewG tog+bm25 (30.26) | +10.00 | +7.50 | +9.27 |

Interpretation:

- Strong positive: MuSiQue across all three models.
- Mild positive: PopQA with DeepSeek and GPT-4o-mini by F1, but accuracy/EM are flat or slightly lower.
- Mixed/negative: PopQA with Gemini when compared against the strongest single baseline.

## NewG Variants vs Paired Retriever Baselines

Each NewG variant is compared with the baseline that supplies its graph retriever family: HippoRAG, ToG, or RAPTOR.

| Dataset | Model | Wins | Avg Delta F1 | Worst Delta | Best Delta |
| --- | --- | --- | --- | --- | --- |
| PopQA | DeepSeek | 4/6 | +3.31 | -0.36 | +9.63 |
| PopQA | GPT-4o-mini | 3/6 | +2.30 | -0.89 | +8.17 |
| PopQA | Gemini | 4/6 | +6.62 | -1.28 | +14.00 |
| MuSiQue | DeepSeek | 4/6 | +3.08 | -5.10 | +12.73 |
| MuSiQue | GPT-4o-mini | 5/6 | +4.66 | -2.67 | +7.58 |
| MuSiQue | Gemini | 5/6 | +8.72 | -5.22 | +22.55 |

Interpretation:

- This is a good positive summary because most dataset/model combinations have 4 or 5 wins out of 6.
- It is stronger than claiming every individual NewG variant wins.

## PopQA Incremental Ablation

This is the older PopQA ablation behind `popqa_ablation_*`.

| Variant | Accuracy | EM | F1 | Narrative |
| --- | --- | --- | --- | --- |
| Simple | 66.00 | 24.00 | 27.15 | reference |
| +Router | 66.00 (+0.00) | 24.00 (+0.00) | 27.15 (+0.00) | mixed |
| +Re-Generator | 66.00 (+0.00) | 19.50 (-4.50) | 25.92 (-1.23) | mixed |
| +Critic & Commendor | 66.50 (+0.50) | 20.00 (-4.00) | 26.05 (-1.10) | mixed |
| w/o Commendor | 58.50 (-7.50) | 57.50 (+33.50) | 27.28 (+0.13) | mixed |
| Single judge/voter | 60.00 (-6.00) | 58.50 (+34.50) | 27.81 (+0.66) | mixed |
| Full NewG | 59.50 (-6.50) | 58.50 (+34.50) | 27.50 (+0.35) | mixed |

Interpretation:

- It is not a clean monotonic module-addition story.
- It does show that answer-control variants greatly increase EM/precision, but accuracy drops.
- Prefer appendix unless the paper explicitly discusses the accuracy-EM tradeoff.

## Leave-One-Out Ablation: Component Contribution

Values are `Full NewG - ablated variant`; positive means removing the component hurts.

| Dataset | Component | Acc contribution | EM contribution | F1 contribution |
| --- | --- | --- | --- | --- |
| PopQA | Router | -0.50 | +0.50 | -0.14 |
| PopQA | Re-generator | -1.50 | -2.00 | -0.63 |
| PopQA | Critic | +2.50 | +2.00 | +1.04 |
| PopQA | Commendor | +1.00 | +1.00 | +0.22 |
| PopQA | Normalizer | -7.00 | +38.50 | +1.45 |
| PopQA | Disambiguation | -0.50 | -0.50 | -0.08 |
| PopQA | Multi-agent voting | -0.50 | +0.00 | -0.31 |
| MuSiQue | Router | -3.00 | -2.00 | -3.19 |
| MuSiQue | Re-generator | -1.00 | -1.00 | -0.78 |
| MuSiQue | Critic | +4.50 | +4.00 | +5.33 |
| MuSiQue | Commendor | -0.50 | -1.00 | -1.25 |
| MuSiQue | Normalizer | -3.50 | +6.50 | +3.59 |
| MuSiQue | Disambiguation | +0.00 | +0.00 | +0.00 |
| MuSiQue | Multi-agent voting | +1.00 | +0.50 | +0.90 |

Main-paper-safe messages:

- Critic is consistently useful, especially on MuSiQue.
- Answer normalizer is important for F1/EM stability, even when raw accuracy can move in the opposite direction.
- Router, re-generator, and disambiguation are not consistently positive in this Gemini hippo+bm25 leave-one-out setting.

## Recommended Positive Figure Set

| Figure | Suggested placement |
| --- | --- |
| positive_best_newg_vs_best_baseline_f1.png | main text or appendix |
| positive_musique_best_newg_metric_gains.png | main text |
| positive_best_newg_gain_matrix.png | main text or appendix |
| positive_counterpart_f1_delta_heatmap.png | main text or appendix |
| positive_counterpart_win_count.png | main text or appendix |
| positive_musique_method_f1_heatmap.png | main text |
| positive_leave_one_out_component_contribution.png | appendix / visual summary |
| positive_loo_critic_normalizer_metric_drop.png | main text |
| positive_loo_key_component_f1_drop.png | main text |
| positive_musique_best_f1_slope.png | main text |
| positive_musique_top_methods_ranked.png | main text |
| positive_musique_metric_radar.png | appendix / visual summary |

## Paper Narrative Recommendation

Use MuSiQue as the main positive evidence: NewG shows clear gains on complex multi-hop QA across all three models. Use PopQA as a robustness or near-parity result on simpler factoid QA, not as the main performance claim. For ablations, focus the main text on critic and answer normalizer; put full mixed ablation charts in appendix or present them as limitations.
