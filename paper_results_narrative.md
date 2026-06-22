# Paper Results Narrative

This note turns the finalized TRACE-RAG experiment tables into paper-facing claims,
figure choices, and draft Results text. It uses the curated, internally
consistent tables in `results_summary.tsv` and `figures/positive_showcase/`.

Important caveat: the local raw result files are not yet a complete
reproducibility archive. The paper figures and tables are consistent, but a
final artifact package should still sync all clean 200-row remote score files
and replace the two known bad local raw audit sources.

## Conservative Claim

The strongest paper claim is on MuSiQue. Across all three model backbones,
TRACE-RAG improves over the strongest baseline on complex multi-hop QA. PopQA should
be framed more carefully: it provides near-parity and paired-retriever evidence
on simpler factoid QA, but it is not a clean global win against the strongest
baseline for every model.

## Main Result Summary

On MuSiQue, the best TRACE-RAG variant beats the best baseline for every tested
model:

| Model | Best baseline | Best TRACE-RAG | Delta Acc | Delta EM | Delta F1 |
| --- | --- | --- | ---: | ---: | ---: |
| DeepSeek | ToG (30.06 F1) | TRACE-RAG tog+bm25 (31.44 F1) | +0.50 | +0.50 | +1.38 |
| GPT-4o-mini | HippoRAG (24.90 F1) | TRACE-RAG hippo+bm25 (31.77 F1) | +2.50 | +5.00 | +6.87 |
| Gemini | RAPTOR (20.99 F1) | TRACE-RAG tog+bm25 (30.26 F1) | +10.00 | +7.50 | +9.27 |

Average MuSiQue gain over the strongest baseline:

| Metric | Average gain |
| --- | ---: |
| Accuracy | +4.33 |
| EM | +4.33 |
| F1 | +5.84 |

On PopQA, DeepSeek and GPT-4o-mini show small positive F1 gains over the best
baseline, while Gemini is slightly below the strongest baseline:

| Model | Best baseline | Best TRACE-RAG | Delta Acc | Delta EM | Delta F1 |
| --- | --- | --- | ---: | ---: | ---: |
| DeepSeek | BM25 (28.86 F1) | TRACE-RAG hippo+bm25 (29.78 F1) | +0.00 | -0.50 | +0.92 |
| GPT-4o-mini | BM25 (28.86 F1) | TRACE-RAG hippo+bm25 (29.06 F1) | -1.00 | -0.50 | +0.20 |
| Gemini | HippoRAG (27.85 F1) | TRACE-RAG hippo+bm25 (27.50 F1) | -2.00 | +0.00 | -0.35 |

This supports a restrained PopQA claim: TRACE-RAG remains competitive on simpler
factoid QA and can improve F1 in several settings, but the main positive
performance story should rely on MuSiQue.

## Paired-Retriever Evidence

The paired-retriever comparison asks whether a TRACE-RAG variant improves over the
graph retriever it builds on, rather than over the single strongest baseline.
Across the full 36 paired comparisons, TRACE-RAG wins 25 times (69.4%) with an
average F1 change of +4.78 over all pairs. Among the positive pairs, the average
F1 gain is +7.67.

Largest paired F1 gains:

| Dataset | Model | Pair | Baseline F1 | TRACE-RAG F1 | Delta F1 |
| --- | --- | --- | ---: | ---: | ---: |
| MuSiQue | Gemini | tog+bm25 | 7.71 | 30.26 | +22.55 |
| MuSiQue | Gemini | tog+vdb | 7.71 | 26.94 | +19.23 |
| PopQA | Gemini | tog+bm25 | 11.97 | 25.97 | +14.00 |
| PopQA | Gemini | tog+vdb | 11.97 | 24.71 | +12.74 |
| MuSiQue | DeepSeek | hippo+vdb | 17.28 | 30.01 | +12.73 |
| MuSiQue | DeepSeek | hippo+bm25 | 17.28 | 28.42 | +11.14 |

This is useful because it shows that TRACE-RAG often strengthens the retriever family
it is paired with, especially when a graph-only baseline is weak or unstable.
It should be presented as complementary evidence rather than as a replacement
for the best-baseline comparison.

## Leave-One-Out Ablation

The leave-one-out ablation measures `Full TRACE-RAG - ablated variant`, so positive
values mean removing the component hurts performance.

The cleanest component story is:

| Dataset | Component | Key positive effects |
| --- | --- | --- |
| MuSiQue | Critic | +5.33 F1, +5.50 recall, +5.37 precision, +4.50 accuracy, +4.00 EM |
| MuSiQue | Answer normalizer | +6.50 EM, +5.17 precision, +3.59 F1 |
| PopQA | Answer normalizer | +38.50 EM, +21.46 precision, +1.45 F1 |
| PopQA | Critic | +2.50 accuracy, +2.25 precision, +2.00 EM, +1.04 F1 |

The Critic is consistently useful, especially on MuSiQue. The answer normalizer
is important for EM and precision stability and still improves F1 on both
datasets. Router, re-generator, disambiguation, and voting are more mixed in
the current Gemini hippo+bm25 leave-one-out setting, so they should be discussed
as nuanced or moved to appendix.

## Recommended Figure Plan

Main-paper set:

| Figure | Best use |
| --- | --- |
| `figures/positive_showcase/panel_musique_best_f1_lift.png` | Primary MuSiQue result as a compact F1-only summary. |
| `figures/positive_showcase/panel_musique_best_newg_metric_gains.png` | Alternate MuSiQue panel if the paper wants to show Accuracy, EM, and F1 together. |
| `figures/positive_showcase/showcase_musique_leaderboard.png` | Detailed MuSiQue method comparison by model and paired retriever family. |
| `figures/positive_showcase/panel_largest_positive_paired_retriever_gains.png` | Compact paired-retriever summary focused on the largest positive gains. |
| `figures/positive_showcase/showcase_positive_paired_gain_heatmap.png` | Full paired-retriever delta grid across datasets/models, useful for appendix or supplemental discussion. |
| `figures/positive_showcase/panel_leave_one_out_removal_hurts_f1.png` | Main ablation visual for Critic and Answer normalizer. |
| `figures/positive_showcase/showcase_musique_bm25_vdb_multimetric_radar.png` | Optional dense visual; best when space allows a broad multi-metric comparison. |

Appendix or limitations:

| Asset | Reason |
| --- | --- |
| `results_summary.tsv` | Full 72-row result grid. |
| `results_popqa.md` and `results_musique.md` | Dataset-specific full tables. |
| `figures/ablation_experiments/leave_one_out_ablation/leave_one_out_*.tsv` | Complete leave-one-out rows, including mixed components. |
| `figures/positive_showcase/showcase_popqa_bm25_vdb_multimetric_radar.png` | Useful robustness view, but PopQA is not the main positive claim. |
| `figures/main_experiments/` mixed-result plots | Better suited for appendix and limitations because they include negative cells. |

## Draft Results Text

TRACE-RAG provides the clearest gains on MuSiQue, the more complex multi-hop QA
benchmark. For each model backbone, the best TRACE-RAG configuration improves over
the strongest non-TRACE-RAG baseline. With DeepSeek, TRACE-RAG tog+bm25 improves F1 from
30.06 to 31.44 (+1.38). With GPT-4o-mini, TRACE-RAG hippo+bm25 improves F1 from
24.90 to 31.77 (+6.87). With Gemini, TRACE-RAG tog+bm25 improves F1 from 20.99 to
30.26 (+9.27). Averaged across the three backbones, TRACE-RAG improves MuSiQue by
+5.84 F1, with matching average gains of +4.33 in both accuracy and EM.

The PopQA results are more mixed, which is expected for a simpler factoid QA
setting where sparse retrieval is already strong. TRACE-RAG hippo+bm25 slightly
improves F1 for DeepSeek (+0.92) and GPT-4o-mini (+0.20), but it is slightly
below the strongest Gemini baseline (-0.35 F1). We therefore treat PopQA as a
robustness setting rather than the central evidence for TRACE-RAG's advantage.

To isolate the effect of adding TRACE-RAG on top of a specific graph retriever, we
also compare each TRACE-RAG variant with its paired retriever baseline. TRACE-RAG improves
25 of 36 paired comparisons (69.4%), with an average change of +4.78 F1 across
all pairs and an average +7.67 F1 among positive pairs. The largest gains occur
when TRACE-RAG complements weaker graph-only baselines, such as Gemini ToG on MuSiQue
(+22.55 F1 with BM25 and +19.23 F1 with VDB).

The leave-one-out ablation highlights two components that are most consistently
useful. Removing the Critic hurts MuSiQue by 5.33 F1 and also reduces accuracy,
EM, precision, and recall. Removing the answer normalizer substantially reduces
EM and precision on both datasets and reduces F1 by 3.59 on MuSiQue and 1.45 on
PopQA. Other components show less consistent positive effects in this setting
and should be interpreted as context-dependent.

## Draft Captions

`panel_musique_best_f1_lift.png`: TRACE-RAG improves over the strongest baseline on
MuSiQue across all model backbones. Bars show the F1 lift for the best TRACE-RAG
variant relative to the best non-TRACE-RAG baseline for each model.

`panel_musique_best_newg_metric_gains.png`: Alternate MuSiQue summary showing
the change in Accuracy, EM, and F1 for the best TRACE-RAG variant relative to the
strongest non-TRACE-RAG baseline for each model.

`showcase_musique_leaderboard.png`: MuSiQue method comparison grouped by model
and graph-retriever family. Gray bars show the original graph baseline; colored
bars show the paired TRACE-RAG variants using BM25 or VDB text retrieval.

`panel_largest_positive_paired_retriever_gains.png`: Ranked paired-retriever
F1 gains. The panel highlights the strongest cases where a TRACE-RAG variant
improves over the graph retriever baseline that supplies its graph component.

`showcase_positive_paired_gain_heatmap.png`: Full paired retriever deltas. Each
cell compares a TRACE-RAG variant with the graph retriever baseline that supplies
its graph component; positive values indicate TRACE-RAG improves over that paired
baseline.

`panel_leave_one_out_removal_hurts_f1.png`: Leave-one-out ablation for the most
useful TRACE-RAG components. Positive values indicate that removing the component
reduces F1 relative to the full TRACE-RAG system.

