# Validation Record

On June 14, 2026, `research_backend/evaluate.py` recomputed metrics from two
existing real 200-query ARC-Fuse experiment outputs:

| Dataset | Accuracy | EM | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PopQA | 59.50 | 58.50 | 59.83 | 21.69 | 27.50 |
| MuSiQue | 19.50 | 18.50 | 27.79 | 25.95 | 26.37 |

Configuration:

- historical method ID: `NewG_hipporag_bm25_gemini-2.5-flash-lite`;
- public method name: `ARCFuse_hipporag_bm25_gemini-2.5-flash-lite`;
- 200 records per dataset;
- real DIGIMON retrieval and model-generated answers.

The recomputed metrics exactly match the archived paper result table.

A fresh online one-query run requires a valid replacement API key and a
working DIGIMON experiment environment. The previously embedded key must not be
reused.
