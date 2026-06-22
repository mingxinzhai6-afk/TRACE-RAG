# ARC-Fuse

ARC-Fuse stands for **Adaptive Routing and Critic-guided Evidence Fusion**.
It is an agentic graph-text RAG system built for complex QA, with:

- query understanding and graph/text/hybrid routing;
- graph and text retriever banks;
- reciprocal-rank and evidence fusion;
- multi-agent judge, vote, and inference;
- structured Critic feedback for directed re-retrieval;
- auxiliary error diagnosis and answer normalization.

The project was previously called `NewG` during experimentation. Historical
result directories keep the `NewG_*` prefix to preserve the paper audit trail.

## What This Release Contains

This repository snapshot is organized as a paper-friendly release package:

- `src/arc_fuse/` - standalone reference implementation for offline smoke tests.
- `research_backend/arc_fuse_digimon/` - paper backend used for the reported experiments.
- `configs/` - standalone demo configuration.
- `research_backend/configs/` - experiment and ablation configurations.
- `examples/` - synthetic offline demo inputs.
- `results/` - archived paper result snapshots.
- `docs/` - reproduction, dataset, and source-map notes.
- `tests/` - offline sanity tests.

For a concise file-by-file map, see [docs/SOURCE_MAP.md](docs/SOURCE_MAP.md).

## Quick Start

### 1. Install the standalone package

```bash
python -m pip install -e .
```

### 2. Run the offline demo

```bash
arc-fuse-demo \
  --config configs/arc_fuse.example.json \
  --corpus examples/corpus.jsonl \
  --graph examples/graph.jsonl \
  --questions examples/questions.jsonl \
  --offline
```

Without installation:

```bash
python run_demo.py \
  --config configs/arc_fuse.example.json \
  --corpus examples/corpus.jsonl \
  --graph examples/graph.jsonl \
  --questions examples/questions.jsonl \
  --offline
```

The offline demo validates orchestration only. It does not reproduce the paper
metrics.

## Paper Backend

The full backend requires a compatible DIGIMON checkout and its Python
environment.

```bash
export DIGIMON_ROOT="/path/to/GraphRAG"
export ARC_FUSE_API_KEY="..."
export ARC_FUSE_BASE_URL="https://api.example.com/v1"
export ARC_FUSE_MODEL="gemini-2.5-flash-lite"
python -m pip install -r research_backend/requirements.txt
```

Run one end-to-end smoke test:

```bash
bash research_backend/scripts/run_real_smoke.sh
```

Run the main PopQA and MuSiQue grid:

```bash
LIMIT=200 DATASETS="datasets/Popqa datasets/musique" \
  bash research_backend/scripts/run_main_experiments.sh
```

Run the HippoRAG+BM25 ablations:

```bash
DATASET=datasets/musique LIMIT=200 \
  bash research_backend/scripts/run_ablation.sh
```

Reproduction details are documented in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Dataset Links

The paper uses two benchmarks:

| Dataset | Link |
| --- | --- |
| PopQA | https://hf.co/datasets/akariasai/PopQA |
| MuSiQue | https://hf.co/datasets/dgslibisey/MuSiQue |

The expected local layout is:

```text
datasets/
  Popqa/
    Corpus.json
    Question.json
  musique/
    Corpus.json
    Question.json
```

See [docs/DATASETS.md](docs/DATASETS.md) for the sample ratio notes used in the paper.

## Core Source Map

The most important source files are:

- `arc_fuse_main.py` - convenience entry point for the release package.
- `src/arc_fuse/cli.py` - standalone CLI.
- `src/arc_fuse/pipeline.py` - standalone orchestration.
- `src/arc_fuse/fusion.py` - evidence fusion helpers.
- `src/arc_fuse/retrievers.py` - lightweight retriever adapters.
- `research_backend/arc_fuse_digimon/runner.py` - paper-run launcher.
- `research_backend/arc_fuse_digimon/engine.py` - paper pipeline control flow.
- `research_backend/arc_fuse_digimon/query_understanding.py` - routing logic.
- `research_backend/arc_fuse_digimon/regeneration.py` - judge/vote regeneration.
- `research_backend/arc_fuse_digimon/critic.py` - Critic decisions.
- `research_backend/arc_fuse_digimon/commendor.py` - auxiliary failure diagnosis.
- `research_backend/arc_fuse_digimon/normalizer.py` - answer normalization.

The full source map lives in [docs/SOURCE_MAP.md](docs/SOURCE_MAP.md).

## Citation

If you use ARC-Fuse in academic work, please cite the paper and keep the
historical `NewG` result names unchanged when referring to archived outputs.

The recommended citation metadata template is [CITATION.cff.template](CITATION.cff.template).

## Security And Licensing

Before publishing this repository, review the following files:

- `SECURITY.md`
- `THIRD_PARTY_NOTICES.md`
- `LICENSE`

No API key is stored in this release package.
