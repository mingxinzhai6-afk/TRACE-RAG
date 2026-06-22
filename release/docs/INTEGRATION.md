# Integration Guide

ARC-Fuse separates the orchestration layer from concrete retrieval systems.

## Lightweight Interface

Implement `arc_fuse.interfaces.AsyncLLM`:

```python
class MyLLM:
    async def complete(self, prompt: str) -> str:
        ...
```

Implement `arc_fuse.interfaces.AsyncRetriever`:

```python
class MyRetriever:
    async def retrieve(self, query, action, top_k):
        return [{
            "id": "chunk-42",
            "content": "Evidence text",
            "score": 0.87,
            "rank": 0,
        }]
```

Then instantiate:

```python
from arc_fuse import ArcFusePipeline

pipeline = ArcFusePipeline(
    llm=llm,
    graph_retriever=graph_backend,
    text_retriever=text_backend,
    config=config,
)
result = await pipeline.query(question)
```

## Paper Implementation Mapping

| Paper backend | Standalone equivalent |
| --- | --- |
| `query_understanding.py` | `arc_fuse.pipeline.QueryUnderstanding` |
| `fusion.py` | `arc_fuse.fusion.rrf_fuse` |
| `disambiguation.py` | `arc_fuse.disambiguation.deduplicate_entities` |
| `evidence.py` | `arc_fuse.pipeline.EvidenceFusion` |
| `regeneration.py` | `arc_fuse.pipeline.RegenerationAgent` |
| `critic.py` | `arc_fuse.pipeline.Critic` |
| `commendor.py` | `arc_fuse.pipeline.Commendor` |
| `normalizer.py` | `arc_fuse.pipeline.AnswerNormalizer` |
| `engine.py::ArcFuseEngine` | `arc_fuse.pipeline.ArcFusePipeline` |

The paper backend preserves the actual DIGIMON-facing behavior used in the
experiments. The standalone implementation is easier to inspect and adapt.
