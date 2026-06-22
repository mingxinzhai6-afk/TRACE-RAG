from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .models import QueryAction


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class LexicalRetriever:
    """Small dependency-free TF-IDF retriever for examples and adapters."""

    def __init__(self, documents: Iterable[dict[str, Any]]):
        self.documents = [dict(document) for document in documents]
        self.document_tokens: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()

        for document in self.documents:
            content = f"{document.get('title', '')} {document.get('content', '')}"
            counts = Counter(_tokens(content))
            self.document_tokens.append(counts)
            document_frequency.update(counts.keys())

        size = max(1, len(self.documents))
        self.idf = {
            token: math.log((size + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }

    async def retrieve(
        self,
        query: str,
        action: QueryAction,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_counts = Counter(_tokens(query))
        scored: list[tuple[float, int]] = []
        for index, document_counts in enumerate(self.document_tokens):
            score = sum(
                query_frequency
                * document_counts.get(token, 0)
                * self.idf.get(token, 0.0)
                for token, query_frequency in query_counts.items()
            )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        results: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(scored[:top_k]):
            document = self.documents[index]
            results.append(
                {
                    "id": document.get("id", f"doc-{index}"),
                    "title": document.get("title", ""),
                    "content": document.get("content", ""),
                    "score": score,
                    "rank": rank,
                }
            )
        return results


class GraphTripleRetriever:
    """Token-overlap retriever over synthetic or exported graph triples."""

    def __init__(self, triples: Iterable[dict[str, Any]]):
        self.triples = [dict(triple) for triple in triples]

    async def retrieve(
        self,
        query: str,
        action: QueryAction,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_terms = set(_tokens(query))
        query_terms.update(_tokens(" ".join(action.entities)))
        query_terms.update(_tokens(" ".join(action.relations)))

        scored: list[tuple[float, int]] = []
        for index, triple in enumerate(self.triples):
            source = str(triple.get("source", ""))
            relation = str(triple.get("relation", ""))
            target = str(triple.get("target", ""))
            source_terms = set(_tokens(source))
            relation_terms = set(_tokens(relation.replace("_", " ")))
            target_terms = set(_tokens(target))
            score = (
                2.0 * len(query_terms & source_terms)
                + 1.5 * len(query_terms & relation_terms)
                + 2.0 * len(query_terms & target_terms)
            )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        results: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(scored[:top_k]):
            triple = self.triples[index]
            source = str(triple.get("source", ""))
            relation = str(triple.get("relation", ""))
            target = str(triple.get("target", ""))
            results.append(
                {
                    "id": triple.get("id", f"edge-{index}"),
                    "content": f"{source} --{relation}--> {target}",
                    "entity_name": source,
                    "score": score,
                    "rank": rank,
                }
            )
        return results
