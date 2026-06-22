# Source Map

This repository keeps only the final ARC-Fuse code paths and the supporting
documentation needed to run them.

## Top-Level Files

| File | Purpose |
| --- | --- |
| `README.md` | Repository overview and usage notes. |
| `arc_fuse_main.py` | Paper backend launcher that expects a compatible DIGIMON checkout. |
| `run_demo.py` | Standalone offline demo entry point. |
| `pyproject.toml` | Package metadata for the standalone source. |
| `.env.example` | Environment variable template. |
| `CITATION.cff.template` | Citation metadata template. |
| `LICENSE` | License for the ARC-Fuse source in this repository. |
| `SECURITY.md` | Security disclosure notes. |
| `THIRD_PARTY_NOTICES.md` | Third-party dependency and provenance notes. |
| `CHANGELOG.md` | Project history and cleanup notes. |
| `CONTRIBUTING.md` | Contribution guidelines. |

## Standalone Reference Implementation

| File | Purpose |
| --- | --- |
| `src/arc_fuse/cli.py` | Command-line interface. |
| `src/arc_fuse/pipeline.py` | Offline orchestration and control flow. |
| `src/arc_fuse/fusion.py` | Evidence fusion helpers. |
| `src/arc_fuse/retrievers.py` | Retriever adapters. |
| `src/arc_fuse/disambiguation.py` | Alias and entity disambiguation. |
| `src/arc_fuse/llm.py` | Model adapter layer. |
| `src/arc_fuse/models.py` | Shared dataclasses and typed structures. |
| `src/arc_fuse/prompts.py` | Prompt templates. |
| `src/arc_fuse/json_utils.py` | JSONL utilities. |
| `src/arc_fuse/interfaces.py` | Stable adapter interfaces. |

## Paper Backend

| File | Purpose |
| --- | --- |
| `research_backend/arc_fuse_digimon/runner.py` | Orchestrates the paper pipeline. |
| `research_backend/arc_fuse_digimon/engine.py` | Main control loop. |
| `research_backend/arc_fuse_digimon/query_understanding.py` | Query routing and feature extraction. |
| `research_backend/arc_fuse_digimon/evidence.py` | Evidence assembly helpers. |
| `research_backend/arc_fuse_digimon/fusion.py` | Retrieval fusion. |
| `research_backend/arc_fuse_digimon/regeneration.py` | Judge/vote/infer regeneration. |
| `research_backend/arc_fuse_digimon/critic.py` | Critic decisions. |
| `research_backend/arc_fuse_digimon/commendor.py` | Failure diagnosis. |
| `research_backend/arc_fuse_digimon/normalizer.py` | Final answer normalization. |
| `research_backend/arc_fuse_digimon/disambiguation.py` | Alias resolution. |
| `research_backend/arc_fuse_digimon/bm25.py` | BM25 helper. |
| `research_backend/arc_fuse_digimon/prompts.py` | Paper prompts. |

## Scripts And Configs

| File | Purpose |
| --- | --- |
| `configs/arc_fuse.example.json` | Standalone demo config. |
| `research_backend/configs/` | Paper backend configs. |
| `research_backend/scripts/run_real_smoke.sh` | One-question end-to-end smoke test. |
| `research_backend/scripts/run_main_experiments.sh` | Full main experiment grid. |
| `research_backend/scripts/run_ablation.sh` | Ablation runner. |
| `examples/` | Offline sample corpus, graph, and questions. |
| `docs/` | Reproduction, dataset, and integration notes. |
| `tests/` | Offline sanity tests. |
| `scripts/scan_secrets.py` | Local repository secret scan helper. |
