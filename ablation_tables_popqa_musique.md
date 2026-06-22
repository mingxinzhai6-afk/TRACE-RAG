# PopQA and MuSiQue Ablation Tables

All values are percentages rounded to two decimals. These tables use
`gemini-2.5-flash-lite`, `hipporag`, `bm25`, and `n=200`.

## PopQA Leave-One-Out Ablation

| Experiment | Setting | Accuracy | EM | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full NewG | all modules enabled | 59.50% | 58.50% | 59.83% | 21.69% | 27.50% |
| w/o Router | only disable `use_routing` | 60.00% | 58.00% | 59.83% | 21.85% | 27.64% |
| w/o Re-Generator | only disable `use_regen` | 61.00% | 60.50% | 61.58% | 22.12% | 28.13% |
| w/o Critic | only disable `use_critic` | 57.00% | 56.50% | 57.58% | 20.86% | 26.46% |
| w/o Commendor | only disable `use_commendor` | 58.50% | 57.50% | 58.83% | 21.57% | 27.28% |
| w/o Normalizer | only disable `use_normalizer` | 66.50% | 20.00% | 38.37% | 25.88% | 26.05% |
| Single judge/voter | cost/stability control | 60.00% | 58.50% | 60.08% | 21.94% | 27.81% |

Source: `figures/ablation_experiments/leave_one_out_ablation/leave_one_out_Popqa_gemini-2.5-flash-lite_hipporag_bm25.tsv`.

The full leave-one-out file also contains `w/o Entity Disambiguation`:
60.00% accuracy, 59.00% EM, 60.33% precision, 21.73% recall, 27.58% F1.

## PopQA Incremental Ablation

| Method | Scale | Accuracy | EM | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple | 200 | 66.00% | 24.00% | 42.16% | 26.49% | 27.15% |
| +Router | 200 | 66.00% | 24.00% | 42.16% | 26.49% | 27.15% |
| +Re-Generator | 200 | 66.00% | 19.50% | 37.78% | 25.78% | 25.92% |
| + Critic & Commendor | 200 | 66.50% | 20.00% | 38.38% | 25.88% | 26.05% |
| w/o Commendor | 200 | 58.50% | 57.50% | 58.83% | 21.57% | 27.28% |
| Single judge/voter | 200 | 60.00% | 58.50% | 60.08% | 21.94% | 27.81% |
| Full NewG | 200 | 59.50% | 58.50% | 59.83% | 21.69% | 27.50% |

## MuSiQue Leave-One-Out Ablation

| Experiment | Setting | Accuracy | EM | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full NewG | all modules enabled | 19.50% | 18.50% | 27.79% | 25.95% | 26.37% |
| w/o Router | only disable `use_routing` | 22.50% | 20.50% | 31.17% | 29.14% | 29.56% |
| w/o Re-Generator | only disable `use_regen` | 20.50% | 19.50% | 29.42% | 26.40% | 27.15% |
| w/o Critic | only disable `use_critic` | 15.00% | 14.50% | 22.42% | 20.45% | 21.04% |
| w/o Commendor | only disable `use_commendor` | 20.00% | 19.50% | 29.54% | 26.87% | 27.62% |
| w/o Normalizer | only disable `use_normalizer` | 23.00% | 12.00% | 22.62% | 25.27% | 22.78% |
| Single judge/voter | cost/stability control | 18.50% | 18.00% | 26.58% | 25.20% | 25.47% |

Source: `figures/ablation_experiments/leave_one_out_ablation/leave_one_out_musique_gemini-2.5-flash-lite_hipporag_bm25.tsv`.

The full leave-one-out file also contains `w/o Entity Disambiguation`:
19.50% accuracy, 18.50% EM, 27.79% precision, 25.95% recall, 26.37% F1.

## MuSiQue Incremental Ablation

| Method | Scale | Accuracy | EM | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple | 200 | 16.50% | 14.00% | 20.58% | 20.38% | 20.07% |
| +Router | 200 | 17.00% | 14.50% | 21.52% | 21.68% | 21.22% |
| +Re-Generator | 200 | 18.00% | 9.00% | 17.28% | 20.77% | 17.78% |
| + Critic & Commendor | 200 | 23.00% | 12.00% | 22.62% | 25.27% | 22.78% |
| w/o Commendor | 200 | 20.00% | 19.50% | 29.54% | 26.87% | 27.62% |
| Single judge/voter | 200 | 18.50% | 18.00% | 26.58% | 25.20% | 25.47% |
| Full NewG | 200 | 19.50% | 18.50% | 27.79% | 25.95% | 26.37% |

Source: `handoff_log_2026-05-13.md`, 2026-05-16 server check of completed
MuSiQue old incremental ablation.
