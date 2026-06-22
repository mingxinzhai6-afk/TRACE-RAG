# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

This is **DIGIMON**, a modular GraphRAG framework that decouples graph
construction, retrieval, indexing, and query-time reasoning so different
methods can be compared under one codebase.

The repository also contains **NewG**, an agentic Graph+Text RAG framework with
adaptive routing, Critic-guided iteration, and optional auxiliary diagnosis.

Paper: https://arxiv.org/abs/2503.04338

## Environment Setup

```bash
conda env create -f experiment.yml -n digimon
conda activate digimon
```

The environment is Python 3.10 with PyTorch, FAISS-GPU, Transformers,
LlamaIndex, and the OpenAI SDK.

## Entry Points

There are three distinct entry points.

### 1. `main.py` - Baseline Methods

Runs a single baseline method with a fixed retrieval strategy.

```bash
python main.py -opt Option/Method/HippoRAG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

Available methods: `GR`, `HippoRAG`, `RAPTOR`, `ToG`, `LightRAG`, `AgentG`,
`Dalk`, `KGP`, `GGraphRAG`, `LGraphRAG`.

### 2. `newg_main.py` - NewG Agentic Framework

Runs the primary NewG pipeline with one graph method and one text method fixed
for the run.

```bash
python newg_main.py -opt Option/Method/NewG.yaml \
    -graph_method hipporag \
    -text_method bm25 \
    -dataset_name datasets/Popqa \
    --eval_limit 200
```

- `graph_method`: `hipporag`, `tog`, `raptor`
- `text_method`: `bm25`, `vdb`
- `--eval_limit 0` runs the full dataset

`Option/Method/NewG.yaml` contains all graph sections, text sections, and the
shared NewG hyperparameters (`n_judges`, `n_voters`, `max_rounds`,
`judge_score_threshold`, `critic_use_normalized_answer`, and related toggles).

### 3. `agentic_main.py` - Legacy Agentic Variants

Runs older agentic wrappers such as `AgenticHippoRAG` and the legacy unified
agentic variants. Prefer `newg_main.py` for current NewG work.

## High-Level Architecture

### Config System

Configs are hierarchical:

1. `Option/Config2.yaml` - global defaults
2. `Option/Method/<METHOD>.yaml` - method overrides
3. CLI args such as `-dataset_name`

`Option.Config2.Config.parse()` merges these. In `newg_main.py`,
`build_graph_config()` additionally merges the selected graph section and text
section into the runtime config.

### Core Module Structure

- `Core/GraphRAG.py` - central orchestrator
- `Core/Chunk/` - document chunking
- `Core/Graph/` - graph construction and storage
- `Core/Index/` - vector indexes
- `Core/Query/` - query engines, including NewG
- `Core/Retriever/` - low-level retrieval operators
- `Core/Storage/` - persistence helpers

### NewG Pipeline

The current `NewGEngine.query()` flow is:

1. `QueryUnderstanding` extracts `{domain, entities, relations, selection}`.
2. Retriever Bank executes `graph`, `text`, or `hybrid` retrieval.
3. `EntityDisambiguation` optionally resolves entity aliases.
4. `EvidenceFusion` builds a fused evidence string and candidate answer.
5. `ReGenerationAgent` produces a short answer from the evidence.
6. `CriticModule` evaluates the answer with verdict in `{pass, retrieve_more, revise}`.
7. If Critic says `revise`, regeneration may retry once on the same evidence.
8. If evidence confidence is low, `Commendor` may provide auxiliary diagnosis:
   `wrong_retriever`, `insufficient_evidence`, `poor_generation`, or `pass`.
9. Critic feedback drives the next retrieval query and optional retriever switch
   up to `max_rounds` (default 3).
10. `AnswerNormalizer` formats the final answer and can optionally normalize the
    answer before Critic via `critic_use_normalized_answer`.

Important semantic note:

- Critic is the primary iteration controller.
- Commendor is auxiliary and mainly used for low-confidence or routing-failure
  diagnosis.
- The current default in `NewG.yaml` is `n_judges: 3` and `n_voters: 3`.

Key NewG files:

- `Core/Query/NewGQuery.py` - main engine and round loop
- `Core/Query/QueryUnderstanding.py` - routing logic and heuristic fallback
- `Core/Query/ReGenerationAgent.py` - judge/voter generation module
- `Core/Query/CriticModule.py` - Critic feedback structure and evaluation
- `Core/Query/Commendor.py` - auxiliary diagnosis
- `Core/Query/AnswerNormalizer.py` - final answer normalization
- `Core/Query/HybridFusion.py` - hybrid retrieval fusion
- `Core/Query/EntityDisambiguation.py` - alias resolution
- `Core/Prompt/NewGPrompt.py` - NewG prompts

### Dataset Format

Each dataset directory such as `datasets/Popqa/` contains:

- `Corpus.json` - one JSON line per document with `title` and `context`
- `Question.json` - one JSON line per query with `question` and `answer`

Loaded by `Data.QueryDataset.RAGQueryDataset`.

### Output Structure

Results are typically written to `output/<dataset_name>/<exp_name>/` and may
also be archived under `newg_logs/<exp_name>/`.

- `Results/results.jsonl` - one JSON line per query
- `Metrics/metrics.json` - aggregated metrics
- `Configs/` - copied configs

NewG saves incrementally every 10 queries (`SAVE_EVERY = 10`) in
`run_query_loop()`.

### Logging

- `logs/` - shell redirection logs from batch scripts
- `Core/Common/Logger.py` - repository logger
- NewG prints per-round diagnostics and final summary stats, including
  Commendor decisions, route-source counts, selection counts, and Critic
  verdicts.

## Common Development Commands

### Quick Smoke Test

```bash
python newg_main.py -opt Option/Method/NewG.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/Popqa --eval_limit 5
```

### Full 20-Experiment Suite

```bash
screen -S main
bash run_20_main.sh
# Ctrl+A, D to detach
```

The suite runs 4 baselines and 6 NewG combinations across PopQA and MuSiQue.

### Ablation Studies

```bash
bash run_ablation_popqa.sh
bash run_ablation_musique.sh
```

Ablation configs live in `Option/Method/NewG_abl_*.yaml`.

Current ablation ladder:

1. `NewG_abl_simple.yaml` - fixed graph retrieval + EvidenceFusion candidate.
2. `NewG_abl_routing.yaml` - adds QueryUnderstanding routing.
3. `NewG_abl_regen.yaml` - adds 3-judge / 3-voter ReGenerationAgent.
4. `NewG_abl_critic.yaml` - adds Critic-driven retrieval and Commendor.
5. `NewG_abl_normalizer.yaml` - full NewG with AnswerNormalizer.

Horizontal controls:

- `NewG_abl_no_commendor.yaml` - full NewG without Commendor.
- `NewG_abl_single_agent.yaml` - full NewG with 1 judge and 1 voter.

### Analyze Results

```bash
# Basic log analysis
python analyze_newg_logs.py --log_dir newg_logs --output analysis.json

# Architecture-level analysis (auto-discovers newg_logs/ and output/datasets/)
python analyze_newg_architecture.py --dataset Popqa

# Or use explicit result globs
python analyze_newg_architecture.py --results newg_logs/NewG_*/Results/results.jsonl
```

### Evaluate Partial Results

```bash
python eval_first_n.py --results output/datasets/Popqa/<exp_name>/Results/results.jsonl --dataset datasets/Popqa --n 50
```

### Change LLM Model

Edit `Option/Config2.yaml` or a method YAML:

```yaml
llm:
  api_type: "openai"
  base_url: "https://api.chatanywhere.tech/v1"
  model: "gpt-4o-mini"
  api_key: "..."
```

For local models, set `api_type: "open_llm"`.

## Key Design Decisions

- No unit-test culture: validation is mostly end-to-end benchmark execution.
- Async throughout: query engines and LLM calls are async.
- Graph method is fixed per run: dynamic routing changes retrieval mode, not the
  underlying graph backend.
- Experiment naming is auto-generated by `newg_main.py`.
- Long runs should use `screen`, `nohup`, or equivalent process management.

## Important File Paths

- `Option/Method/NewG.yaml` - primary NewG config
- `Option/Config2.yaml` - global defaults
- `Core/Query/NewGQuery.py` - NewG engine core
- `Core/Query/QueryUnderstanding.py` - routing logic
- `Core/Prompt/NewGPrompt.py` - NewG prompts
- `newg_main.py` - NewG entry point
- `main.py` - baseline entry point
- `run_20_main.sh` - experiment runner
- `analyze_newg_logs.py` - general log analysis
- `analyze_newg_architecture.py` - architecture/control-flow analysis
- `datasets/Popqa/` - primary single-hop benchmark
- `datasets/musique/` - primary multi-hop benchmark
