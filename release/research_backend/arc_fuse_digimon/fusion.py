"""
HybridFusion — Retriever Bank internal fusion.

Fuses ranked lists from Graph Retriever and Text Retriever into a single
ranked list using Reciprocal Rank Fusion (RRF).

This is the "fusion strategy" box INSIDE the Hybrid Retriever. The final
evidence assembly (for the generator) is handled by EvidenceFusion.py.

RRF formula:
    RRF_score(d) = Σ_{L in lists} 1 / (k + rank_L(d))

where k is a constant (default 60, as in the original RRF paper).

Usage:
    fused = rrf_fuse(
        graph_results=[{"id": ..., "content": ..., "rank": ...}, ...],
        text_results =[{"id": ..., "content": ..., "rank": ...}, ...],
        top_k=10,
    )
    # Returns list of dicts with "rrf_score" added and sorted descending.
"""

from typing import List, Dict, Optional
from Core.Common.Logger import logger


DEFAULT_RRF_K = 60


def _doc_key(doc: Dict) -> str:
    """Unique key for a document across lists (prefer id, fall back to content prefix)."""
    for k in ("chunk_id", "id", "entity_name", "src_tgt"):
        v = doc.get(k)
        if v:
            return str(v)
    content = doc.get("content") or doc.get("description") or ""
    return content[:120]


def rrf_fuse(
    graph_results: List[Dict],
    text_results: List[Dict],
    top_k: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
    graph_weight: float = 1.0,
    text_weight: float = 1.0,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion of two ranked lists.

    Args:
        graph_results: ranked list from Graph Retriever, each dict should have
                       a "rank" field (0-based). If absent, uses list position.
        text_results:  ranked list from Text Retriever.
        top_k:         number of fused results to return.
        rrf_k:         RRF constant (default 60).
        graph_weight:  weight multiplier for graph list contributions.
        text_weight:   weight multiplier for text list contributions.

    Returns:
        fused list of dicts, each augmented with:
          "rrf_score": float
          "sources":   list of source tags (e.g., ["graph", "text"])
    """
    if not graph_results and not text_results:
        return []

    pool: Dict[str, Dict] = {}

    for i, doc in enumerate(graph_results or []):
        rank = doc.get("rank", i)
        key = _doc_key(doc)
        contrib = graph_weight / (rrf_k + rank + 1)
        if key not in pool:
            pool[key] = dict(doc)
            pool[key]["rrf_score"] = 0.0
            pool[key]["sources"] = []
        pool[key]["rrf_score"] += contrib
        if "graph" not in pool[key]["sources"]:
            pool[key]["sources"].append("graph")

    for i, doc in enumerate(text_results or []):
        rank = doc.get("rank", i)
        key = _doc_key(doc)
        contrib = text_weight / (rrf_k + rank + 1)
        if key not in pool:
            pool[key] = dict(doc)
            pool[key]["rrf_score"] = 0.0
            pool[key]["sources"] = []
        pool[key]["rrf_score"] += contrib
        if "text" not in pool[key]["sources"]:
            pool[key]["sources"].append("text")

    fused = sorted(pool.values(), key=lambda d: d["rrf_score"], reverse=True)
    return fused[:top_k]


def format_fused_as_string(fused: List[Dict], max_chars: int = 4000) -> str:
    """Render fused results as a readable string block for the generator."""
    if not fused:
        return ""

    lines = []
    total = 0
    for i, doc in enumerate(fused):
        src = "+".join(doc.get("sources", []))
        content = (doc.get("content")
                   or doc.get("description")
                   or doc.get("entity_name")
                   or str(doc))
        score = doc.get("rrf_score", 0.0)
        snippet = f"[{i+1}] ({src}, rrf={score:.4f}) {content}"
        if total + len(snippet) > max_chars:
            break
        lines.append(snippet)
        total += len(snippet)
    return "\n\n".join(lines)


def normalize_graph_output(raw_graph_output) -> List[Dict]:
    """
    Coerce graph retriever output (str, list of str, or list of dict) into
    a ranked list of dicts with "rank", "content", "id" fields.

    This allows HybridFusion to accept the heterogeneous outputs of
    PPRQuery / ToGQuery / BasicQuery uniformly.
    """
    if raw_graph_output is None:
        return []

    if isinstance(raw_graph_output, str):
        # Split on double-newline / "->" separators
        parts = [p.strip() for p in raw_graph_output.split("\n\n") if p.strip()]
        if not parts:
            parts = [p.strip() for p in raw_graph_output.split("\n") if p.strip()]
        return [
            {"id": f"g{i}", "content": p, "rank": i}
            for i, p in enumerate(parts)
        ]

    if isinstance(raw_graph_output, (list, tuple)):
        results = []
        for i, item in enumerate(raw_graph_output):
            if isinstance(item, dict):
                d = dict(item)
                d.setdefault("rank", i)
                d.setdefault("content", item.get("description") or item.get("entity_name") or str(item))
                d.setdefault("id", f"g{i}")
                results.append(d)
            else:
                results.append({"id": f"g{i}", "content": str(item), "rank": i})
        return results

    logger.warning(f"HybridFusion: unexpected graph output type {type(raw_graph_output)}")
    return [{"id": "g0", "content": str(raw_graph_output), "rank": 0}]


def normalize_text_output(raw_text_output) -> List[Dict]:
    """Coerce BM25Retriever or VDB retriever output to a ranked list of dicts."""
    if raw_text_output is None:
        return []

    if isinstance(raw_text_output, list):
        results = []
        for i, item in enumerate(raw_text_output):
            if isinstance(item, dict):
                d = dict(item)
                d.setdefault("rank", i)
                d.setdefault("content", item.get("content") or str(item))
                d.setdefault("id", item.get("chunk_id") or f"t{i}")
                results.append(d)
            else:
                results.append({"id": f"t{i}", "content": str(item), "rank": i})
        return results

    if isinstance(raw_text_output, str):
        parts = [p.strip() for p in raw_text_output.split("\n\n") if p.strip()]
        return [
            {"id": f"t{i}", "content": p, "rank": i}
            for i, p in enumerate(parts)
        ]

    return []
