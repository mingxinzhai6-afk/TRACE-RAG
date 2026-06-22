from Core.Common.Constants import Retriever
from Core.Common.Logger import logger
from Core.Query.BasicQuery import BasicQuery
from Core.Retriever.BM25Retriever import BM25Retriever


class _TextOnlyQueryMixin:
    """Shared helpers for text-only baseline query methods."""

    def _top_k(self) -> int:
        return int(getattr(self.config, "top_k", 10) or 10)

    def _max_chars(self) -> int:
        token_budget = int(
            getattr(self.config, "local_max_token_for_text_unit", 4000) or 4000
        )
        return token_budget * 4

    @staticmethod
    def _retrieval_query(query: str) -> str:
        query = str(query or "").strip()
        marker = "Answer with a short factual span"
        if marker in query:
            return query.split(marker, 1)[0].strip()
        return query.split("\n", 1)[0].strip() or query

    @staticmethod
    def _format_passages(passages, source: str, max_chars: int) -> str:
        if not passages:
            return ""

        rows = ["-----Sources-----", "```csv", "id,source,score,content"]
        total = sum(len(row) for row in rows)
        for i, passage in enumerate(passages):
            if isinstance(passage, dict):
                content = (
                    passage.get("content")
                    or passage.get("description")
                    or passage.get("entity_name")
                    or ""
                )
                score = passage.get("score", passage.get("rrf_score", ""))
            else:
                content = str(passage)
                score = ""
            content = str(content).replace('"', '""').replace("\n", " ")
            row = f'{i},"{source}","{score}","{content}"'
            if total + len(row) > max_chars:
                break
            rows.append(row)
            total += len(row)
        rows.append("```")
        return "\n".join(rows)


class BM25Query(_TextOnlyQueryMixin, BasicQuery):
    """Pure lexical BM25 baseline over DIGIMON chunks."""

    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)
        doc_chunk = self._retriever.context.as_dict.get("doc_chunk")
        self._bm25 = BM25Retriever(doc_chunk=doc_chunk, top_k=self._top_k())

    async def _retrieve_relevant_contexts(self, query):
        retrieval_query = self._retrieval_query(query)
        passages = await self._bm25.retrieve(retrieval_query, top_k=self._top_k())
        logger.info(f"BM25 baseline retrieved {len(passages)} chunks")
        return self._format_passages(passages, "bm25", self._max_chars())


class VDBQuery(_TextOnlyQueryMixin, BasicQuery):
    """
    Entity-VDB text baseline.

    This mirrors NewG's text_method=vdb control: retrieve entities from the
    entity vector index, then use entity occurrence links to fetch text chunks.
    """

    async def _retrieve_relevant_contexts(self, query):
        retrieval_query = self._retrieval_query(query)
        node_datas = await self._retriever.retrieve_relevant_content(
            type=Retriever.ENTITY,
            mode="vdb",
            seed=retrieval_query,
            top_k=self._top_k(),
        )
        if not node_datas:
            logger.info("VDB baseline retrieved 0 entities")
            return ""

        text_units = await self._retriever.retrieve_relevant_content(
            type=Retriever.CHUNK,
            mode="entity_occurrence",
            node_datas=node_datas,
        )
        text_units = text_units or []
        passages = [
            {"content": text, "score": 1.0 / (i + 1)}
            for i, text in enumerate(text_units[: self._top_k()])
        ]
        logger.info(
            f"VDB baseline retrieved {len(node_datas)} entities and "
            f"{len(passages)} chunks"
        )
        return self._format_passages(passages, "vdb", self._max_chars())
