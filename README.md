# ARC-Fuse

ARC-Fuse stands for **Adaptive Routing and Critic-guided Evidence Fusion**.
It is an agentic graph-text RAG system for complex QA, with:

- query understanding and graph/text/hybrid routing;
- graph and text retriever banks;
- reciprocal-rank and evidence fusion;
- multi-agent judge, vote, and inference;
- structured Critic feedback for directed re-retrieval;
- auxiliary error diagnosis and answer normalization.

This repository keeps the final ARC-Fuse codebase only. Historical GraphRAG/NewG
experiment trees and large generated artifacts are intentionally omitted from GitHub.

## Repository Layout

- `src/arc_fuse/` - standalone reference implementation.
- `research_backend/arc_fuse_digimon/` - paper backend used for the reported experiments.
- `arc_fuse_main.py` - convenience entry point for the paper backend.
- `run_demo.py` - standalone offline demo entry point.
- `configs/` - demo configuration.
- `research_backend/configs/` - paper backend runtime configuration.
- `examples/` - offline sample inputs.
- `docs/` - reproduction, dataset, and source-map notes.
- `tests/` - offline sanity tests.
- `CHANGELOG.md` - project history and cleanup notes.
- `CONTRIBUTING.md` - contribution guidelines.

For a concise file-by-file map, see [docs/SOURCE_MAP.md](docs/SOURCE_MAP.md).

## Quick Start

### Standalone demo

```bash
python -m pip install -e .
arc-fuse-demo \
  --config configs/arc_fuse.example.json \
  --corpus examples/corpus.jsonl \
  --graph examples/graph.jsonl \
  --questions examples/questions.jsonl \
  --offline
```

Or run without installing:

```bash
python run_demo.py \
  --config configs/arc_fuse.example.json \
  --corpus examples/corpus.jsonl \
  --graph examples/graph.jsonl \
  --questions examples/questions.jsonl \
  --offline
```

The offline demo validates orchestration only. It does not reproduce paper metrics.

### Paper backend

The paper backend requires a compatible DIGIMON checkout and Python environment.

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

## Citation

If you use ARC-Fuse in academic work, please cite the paper.

The recommended citation metadata template is [CITATION.cff.template](CITATION.cff.template).

## Security And Licensing

Before publishing this repository, review:

- `SECURITY.md`
- `THIRD_PARTY_NOTICES.md`
- `LICENSE`

No API key is stored in this repository.
