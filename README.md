# TRACE-RAG

**Traceable Retrieval with Adaptive Critic and Evidence-Enhanced Generation**

TRACE-RAG is an adaptive, graph-text agentic RAG framework for complex
question answering. It combines query understanding, dynamic retrieval
routing, graph/text evidence fusion, multi-judge answer regeneration, Critic
feedback, failure attribution, and answer normalization in one iterative
closed loop.

> This repository contains the reproducible implementation and experiment
> backend for the TRACE-RAG paper. The standalone package is named `arc-fuse`
> for historical compatibility; it implements the same method and is referred
> to as TRACE-RAG in the paper.

## Method overview

For each question, TRACE-RAG:

1. parses the question and extracts entities, relations, and temporal cues;
2. routes the query to graph, text, or hybrid retrieval;
3. fuses heterogeneous evidence and resolves entity aliases;
4. generates a candidate answer with a multi-judge
   `dispatch -> vote -> evidence-grounded inference` process;
5. validates the answer with a Critic, attributes failures, and either accepts,
   regenerates, or retrieves more evidence;
6. normalizes the final answer for reliable exact-match evaluation.

The graph branch supports HippoRAG, ToG, and RAPTOR; the text branch supports
BM25 and VDB. This design makes retrieval decisions and answer corrections
explicit rather than treating the first retrieved context as final.

## Quick start: offline demo

The offline demo checks the orchestration without calling an external model or
reproducing the paper's reported metrics.

```bash
python -m pip install -e .
arc-fuse-demo \
  --config configs/arc_fuse.example.json \
  --corpus examples/corpus.jsonl \
  --graph examples/graph.jsonl \
  --questions examples/questions.jsonl \
  --offline
```

Alternatively, run it directly from the repository:

```bash
python run_demo.py \
  --config configs/arc_fuse.example.json \
  --corpus examples/corpus.jsonl \
  --graph examples/graph.jsonl \
  --questions examples/questions.jsonl \
  --offline
```

## Reproducing the paper experiments

The paper backend is in `research_backend/arc_fuse_digimon/` and uses a
DIGIMON-compatible GraphRAG checkout as an external dependency.

```bash
python -m pip install -r research_backend/requirements.txt

export DIGIMON_ROOT="/path/to/GraphRAG"
export ARC_FUSE_API_KEY="<your-api-key>"
export ARC_FUSE_BASE_URL="https://api.example.com/v1"
export ARC_FUSE_MODEL="gemini-2.5-flash-lite"
```

Run a real end-to-end smoke test:

```bash
bash research_backend/scripts/run_real_smoke.sh
```

Run the 200-question PopQA and MuSiQue experiment grid:

```bash
LIMIT=200 DATASETS="datasets/Popqa datasets/musique" \
  bash research_backend/scripts/run_main_experiments.sh
```

The grid evaluates six graph-text combinations for each dataset:

`hipporag+bm25`, `hipporag+vdb`, `tog+bm25`, `tog+vdb`,
`raptor+bm25`, and `raptor+vdb`.

For full environment, data layout, output, and evaluator instructions, see
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Datasets and evaluation

The paper evaluates 200 consistently selected examples from each dataset:

- [PopQA](https://huggingface.co/datasets/akariasai/PopQA), for factual and
  short-answer robustness;
- [MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue), for multi-hop
  evidence composition and iterative retrieval.

Expected local layout:

```text
datasets/
├── Popqa/
│   ├── Corpus.json
│   └── Question.json
└── musique/
    ├── Corpus.json
    └── Question.json
```

Results are reported with Accuracy, Exact Match (EM), Precision, Recall, and
F1. The paper also includes fixed-baseline combination experiments and
leave-one-out ablations for Router, ReGenerationAgent, Critic, failure
attribution, and AnswerNormalizer.

## Recreating paper figures

The plotting source for the experiment-result figures is included in
[`scripts/draw_paper_result_figures.py`](scripts/draw_paper_result_figures.py).
Install the plotting dependencies and run:

```bash
python -m pip install -r requirements-figures.txt
python scripts/draw_paper_result_figures.py
```

Figures are written to `figures/generated_results/`. The default input table
for the paired-retriever heatmap is
`data/plotting/paired_retriever_deltas.tsv`; use `--output-dir` and
`--paired-data` to select another output directory or result table. The script
recreates the MuSiQue metric-gain, F1-lift, main-experiment, paired-retriever,
and leave-one-out figures used in the paper.

## Repository layout

- `src/arc_fuse/` — standalone TRACE-RAG orchestration and CLI.
- `research_backend/arc_fuse_digimon/` — paper experiment integration.
- `research_backend/scripts/` — smoke-test and main-grid launchers.
- `configs/` and `research_backend/configs/` — demo and experiment settings.
- `datasets/` — local dataset files (not bundled in full).
- `docs/` — reproduction and dataset notes.
- `figures/` — paper figures, including the editable Visio architecture source.
- `data/plotting/` — compact tabular inputs used by the result-figure script.
- `requirements-figures.txt` — plotting-only Python dependencies.
- `tests/` — lightweight checks for the standalone implementation.

## Paper and citation

The paper title is **TRACE-RAG: A Framework for Traceable Retrieval with
Adaptive Critic and Evidence-Enhanced Generation**. The Chinese manuscript is
available in [`paper_cn_full.tex`](paper_cn_full.tex). The paper reports that
TRACE-RAG obtains its most consistent gains on MuSiQue, while the ablations
identify graph-text fusion, Critic feedback, and answer normalization as key
contributors.

If you use this repository, please cite the paper. Citation metadata can be
prepared from [`CITATION.cff.template`](CITATION.cff.template).

## License and security

See [`LICENSE`](LICENSE), [`SECURITY.md`](SECURITY.md), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). API keys must be supplied
through environment variables; do not commit credentials or generated private
datasets.
