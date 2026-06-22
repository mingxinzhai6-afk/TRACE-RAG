# NewG Leave-One-Out Ablation Commands

This is the replacement ablation design for the PopQA and MuSiQue module study.
It uses Full NewG as the reference and disables exactly one module at a time.

## Variants

```text
full
no_router
no_regen
no_critic
no_commendor
no_normalizer
no_disambiguation
single_agent
```

## Already-Covered Variants

The runner skips completed equivalent results by default (`SKIP_COMPLETED=1`):

| Leave-one-out variant | Reused result if complete |
|---|---|
| `full` | `NewG_hipporag_bm25_gemini-2.5-flash-lite` |
| `no_commendor` | `NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_no_commendor` |
| `no_normalizer` | `NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_critic` |
| `single_agent` | `NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_single_agent` |

The old `Simple`, `+Router`, and `+Re-Generator` incremental runs are not reused
as leave-one-out results because they disable multiple modules at once.

## Run Both Datasets

Run this only after the current MuSiQue ablation process is idle:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
screen -S loo_both
bash run_leave_one_out_ablation_both_datasets.sh
```

Equivalent one-line detached start:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
nohup bash run_leave_one_out_ablation_both_datasets.sh > logs/loo_both.nohup.log 2>&1 &
```

## Run One Dataset

PopQA:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
DATA=datasets/Popqa bash run_leave_one_out_ablation.sh
```

MuSiQue:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
DATA=datasets/musique bash run_leave_one_out_ablation.sh
```

## Run Only Truly New Variants

This is equivalent in practice when the old completed results are present, but
is useful when you want the command to explicitly list only missing experiment
types:

```bash
VARIANTS="no_router no_regen no_critic no_disambiguation" \
bash run_leave_one_out_ablation_both_datasets.sh
```

## Collect And Plot

The both-dataset runner automatically collects and plots after each dataset.
Manual commands:

```bash
python collect_leave_one_out_ablation.py --dataset datasets/Popqa
python plot_leave_one_out_ablation.py --dataset datasets/Popqa

python collect_leave_one_out_ablation.py --dataset datasets/musique
python plot_leave_one_out_ablation.py --dataset datasets/musique
```

Outputs are written under `figures/` with prefixes like:

```text
leave_one_out_Popqa_gemini-2.5-flash-lite_hipporag_bm25
leave_one_out_musique_gemini-2.5-flash-lite_hipporag_bm25
```
