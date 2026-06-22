# Paper Reproduction

## Name Mapping

ARC-Fuse was called `NewG` while the experiments were running. The following
identifiers refer to the same method:

| Historical result prefix | Public name |
| --- | --- |
| `NewG_hipporag_bm25_*` | `ARCFuse_hipporag_bm25_*` |
| `NewG_hipporag_vdb_*` | `ARCFuse_hipporag_vdb_*` |
| `NewG_tog_bm25_*` | `ARCFuse_tog_bm25_*` |
| `NewG_tog_vdb_*` | `ARCFuse_tog_vdb_*` |
| `NewG_raptor_bm25_*` | `ARCFuse_raptor_bm25_*` |
| `NewG_raptor_vdb_*` | `ARCFuse_raptor_vdb_*` |

Do not rename historical directories. New runs use the `ARCFuse_*` prefix.

## DIGIMON Dependency

The experiment code uses the DIGIMON repository:

```text
https://github.com/JayLZhou/GraphRAG
```

The integration was checked against upstream HEAD:

```text
4e87938e46f90f3616fb27f955e8b2dc43743bde
```

The original experiments were run in a locally modified DIGIMON-derived
workspace without Git metadata. `research_backend/compatibility_manifest.json`
records hashes of the critical local backend files for audit purposes.

The upstream revision above has no license file. ARC-Fuse therefore imports
DIGIMON as an external dependency and does not redistribute its source.

## Dataset Layout

Place datasets under the DIGIMON root:

```text
datasets/
  Popqa/
    Corpus.json
    Question.json
  musique/
    Corpus.json
    Question.json
```

Each file is JSON Lines. Corpus rows contain `title` and `context`; question
rows contain `question` and `answer`.

See `docs/DATASETS.md` for the official dataset links and the 200-sample
paper subset ratios.

## Environment

Create the DIGIMON environment according to its own instructions, then install
the ARC-Fuse additions:

```bash
python -m pip install -r research_backend/requirements.txt
export DIGIMON_ROOT="/path/to/GraphRAG"
export ARC_FUSE_API_KEY="..."
export ARC_FUSE_BASE_URL="https://api.example.com/v1"
export ARC_FUSE_MODEL="gemini-2.5-flash-lite"
```

`ARC_FUSE_BASE_URL` and `ARC_FUSE_MODEL` override the YAML defaults.

## Real Smoke Test

```bash
bash research_backend/scripts/run_real_smoke.sh
```

This runs one PopQA query with HippoRAG+BM25 through the real graph
construction, retrieval, generation, Critic, and normalization pipeline.

## Main Grid

```bash
LIMIT=200 DATASETS="datasets/Popqa datasets/musique" \
  bash research_backend/scripts/run_main_experiments.sh
```

The grid contains:

- graph: `hipporag`, `tog`, `raptor`;
- text: `bm25`, `vdb`;
- six ARC-Fuse combinations per dataset.

## Ablations

```bash
DATASET=datasets/musique LIMIT=200 \
  bash research_backend/scripts/run_ablation.sh
```

Included configurations:

- simple generation;
- adaptive routing;
- multi-agent re-generation;
- Critic and Commendor;
- full ARC-Fuse;
- no Commendor;
- one judge and one voter.

## Outputs

New outputs are written below the DIGIMON workspace:

```text
output/<dataset>/ARCFuse_<graph>_<text>_<model>/
  Results/results.jsonl
  Results/metrics.json
  Configs/
  Metrics/
```

The standalone evaluator can be rerun with:

```bash
python research_backend/evaluate.py \
  --result_path /path/to/results.jsonl \
  --limit 200
```

Reported paper metrics are archived in
`results/legacy_newg_results_summary.tsv`. The filename intentionally retains
the historical name.
