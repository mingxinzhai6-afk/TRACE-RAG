"""
BM25 text retriever used by ARC-Fuse experiments.

Builds a BM25 index over all chunks in the DIGIMON chunk storage (doc_chunk),
and returns ranked passages for a given query.

Unlike ChunkRetriever which depends on graph entities for retrieval modes
(ppr / entity_occurrence / from_relation), BM25Retriever does pure lexical
text retrieval directly from the chunk corpus.

Usage:
    retriever = BM25Retriever(doc_chunk=digimon.doc_chunk, top_k=10)
    await retriever.build_index()
    passages = await retriever.retrieve(query, top_k=10)
    # Returns: [{"chunk_id": ..., "content": ..., "score": ...}, ...]
"""

import re
import asyncio
from typing import List, Dict, Optional
from Core.Common.Logger import logger

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    BM25Okapi = None
    _HAS_BM25 = False


# Simple English tokenizer — lowercase, split on non-alphanumeric, drop short tokens.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


class BM25Retriever:
    """
    Pure BM25 text retriever over DIGIMON's chunk storage.

    The index is built once (lazily) on the first call to build_index(),
    then cached for the lifetime of the instance.
    """

    def __init__(self, doc_chunk, top_k: int = 10):
        if not _HAS_BM25:
            raise ImportError(
                "BM25Retriever requires 'rank_bm25'. "
                "Install with: pip install rank_bm25"
            )
        self.doc_chunk = doc_chunk
        self.top_k = top_k
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: List[str] = []
        self._chunk_texts: List[str] = []
        self._built = False

    async def build_index(self):
        """Build BM25 index by iterating all chunks in doc_chunk storage."""
        if self._built:
            return

        logger.info("BM25Retriever: building index over all chunks...")
        chunk_ids, chunk_texts = await self._collect_chunks()

        if not chunk_texts:
            logger.warning("BM25Retriever: no chunks found in storage")
            self._built = True
            return

        tokenized_corpus = [_tokenize(t) for t in chunk_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunk_ids = chunk_ids
        self._chunk_texts = chunk_texts
        self._built = True
        logger.info(f"BM25Retriever: indexed {len(chunk_texts)} chunks")

    async def _collect_chunks(self):
        """
        Extract (chunk_id, text) pairs from the doc_chunk storage.

        doc_chunk is a DocChunk instance; doc_chunk._chunk is a ChunkKVStorage
        whose public async get_chunks() returns list[(chunk_id, TextChunk)].
        TextChunk.content holds the text.
        """
        chunk_ids: List[str] = []
        chunk_texts: List[str] = []

        # doc_chunk._chunk is ChunkKVStorage; use its public async API
        chunk_store = getattr(self.doc_chunk, "_chunk", None)
        if chunk_store is None:
            # doc_chunk might itself be ChunkKVStorage
            chunk_store = self.doc_chunk

        try:
            items = await chunk_store.get_chunks()  # list of (chunk_id_str, TextChunk)
        except Exception as e:
            logger.warning(f"BM25Retriever: get_chunks() failed: {e}, trying internal dict")
            raw = getattr(chunk_store, "_chunk", {})
            items = list(raw.items()) if isinstance(raw, dict) else []

        for cid, tc in items:
            text = getattr(tc, "content", None) or getattr(tc, "text", None)
            if text is None and isinstance(tc, dict):
                text = tc.get("content") or tc.get("text")
            if text:
                chunk_ids.append(str(cid))
                chunk_texts.append(text)

        return chunk_ids, chunk_texts

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve top-k chunks ranked by BM25 score for the given query.

        Returns:
            list of dicts: {"chunk_id", "content", "score", "rank"}
            Sorted by score descending.
        """
        if not self._built:
            await self.build_index()

        if self._bm25 is None or not self._chunk_texts:
            return []

        k = top_k if top_k is not None else self.top_k
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # argsort descending
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                "chunk_id": self._chunk_ids[idx],
                "content": self._chunk_texts[idx],
                "score": score,
                "rank": rank,
            })
        return results

    async def retrieve_as_string(self, query: str, top_k: Optional[int] = None,
                                  max_chars: int = 4000) -> str:
        """Retrieve and format top-k passages as a single reference string."""
        passages = await self.retrieve(query, top_k)
        if not passages:
            return ""

        buf = []
        total = 0
        for p in passages:
            snippet = f"[BM25 rank {p['rank'] + 1}, score={p['score']:.2f}] {p['content']}"
            if total + len(snippet) > max_chars:
                break
            buf.append(snippet)
            total += len(snippet)
        return "\n\n".join(buf)
