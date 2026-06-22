from __future__ import annotations

from typing import Any, Mapping, Sequence


def _document_key(document: Mapping[str, Any]) -> str:
    for field in ("id", "chunk_id", "entity_name", "src_tgt"):
        value = document.get(field)
        if value:
            return str(value)
    return str(document.get("content", ""))[:160]


def normalize_results(
    raw_results: Sequence[Mapping[str, Any]] | Sequence[str] | str | None,
    prefix: str,
) -> list[dict[str, Any]]:
    if raw_results is None:
        return []
    if isinstance(raw_results, str):
        items: Sequence[Any] = [
            part.strip() for part in raw_results.split("\n\n") if part.strip()
        ]
    else:
        items = raw_results

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            document = dict(item)
            document.setdefault("id", f"{prefix}-{index}")
            document.setdefault("rank", index)
            document.setdefault(
                "content",
                document.get("description")
                or document.get("entity_name")
                or str(item),
            )
        else:
            document = {
                "id": f"{prefix}-{index}",
                "rank": index,
                "content": str(item),
            }
        normalized.append(document)
    return normalized


def rrf_fuse(
    graph_results: Sequence[Mapping[str, Any]],
    text_results: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    rrf_k: int = 60,
    graph_weight: float = 1.0,
    text_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Fuse graph and text rankings with reciprocal-rank fusion."""
    pool: dict[str, dict[str, Any]] = {}
    inputs = (
        ("graph", graph_results, graph_weight),
        ("text", text_results, text_weight),
    )
    for source, results, weight in inputs:
        for position, raw_document in enumerate(results):
            document = dict(raw_document)
            key = _document_key(document)
            rank = int(document.get("rank", position))
            entry = pool.setdefault(
                key,
                {
                    **document,
                    "rrf_score": 0.0,
                    "sources": [],
                },
            )
            entry["rrf_score"] += weight / (rrf_k + rank + 1)
            if source not in entry["sources"]:
                entry["sources"].append(source)

    return sorted(
        pool.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:top_k]
