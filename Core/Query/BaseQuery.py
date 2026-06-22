import asyncio
from abc import ABC, abstractmethod
from Core.Retriever.MixRetriever import MixRetriever
from typing import Any
from Core.Prompt import GraphPrompt, QueryPrompt
from Core.Common.Utils import clean_str, prase_json_from_response, truncate_list_by_token_size, \
    list_to_quoted_csv_string
from Core.Common.Logger import logger


class BaseQuery(ABC):
    def __init__(self, config, retriever_context):
        self._retriever = MixRetriever(retriever_context)
        self.config = config
        self.llm = self._retriever.llm
        # Lazy-init agentic modules (only when enabled in config)
        self._normalizer = None
        self._critic = None

    def _get_normalizer(self):
        if self._normalizer is None:
            from Core.Query.AnswerNormalizer import AnswerNormalizer
            self._normalizer = AnswerNormalizer(self.llm)
        return self._normalizer

    def _get_critic(self):
        if self._critic is None:
            from Core.Query.CriticModule import CriticModule
            self._critic = CriticModule(self.llm)
        return self._critic

    @abstractmethod
    async def _retrieve_relevant_contexts(self, **kwargs):
        pass

    async def query(self, query):
        context = await self._retrieve_relevant_contexts(query=query)
        response = None
        if self.config.query_type == "summary":
            response = await self.generation_summary(query, context)
        elif self.config.query_type == "qa":
            response = await self.generation_qa(query, context)
        else:
            logger.error("Invalid query type")
            return response

        # ---- Innovation 1: Critic-driven Directed Subgraph Expansion ----
        if getattr(self.config, 'use_critic', False) and response:
            response, context = await self._critic_loop(query, context, response)

        # ---- Innovation 2: Answer Normalizer ----
        if getattr(self.config, 'use_answer_normalizer', False) and response:
            response = await self._normalize_answer(query, response)

        return response

    async def _critic_loop(self, query: str, context, response: str):
        """
        Critic-driven iterative refinement loop.

        critic_mode="directed" (default, our method):
            Structured feedback → specific retrieval actions:
            missing_entity → expand neighbors, broken_path → complete path,
            conflict → disambiguate. Targeted re-retrieval with enriched query.

        critic_mode="blind" (ablation baseline, Agent-G style):
            Critic validates only (pass/fail). On failure, re-retrieves with
            the original query unchanged — no directed guidance.
        """
        critic = self._get_critic()
        max_rounds = getattr(self.config, 'critic_max_rounds', 2)
        critic_mode = getattr(self.config, 'critic_mode', 'directed')
        ctx_str = context if isinstance(context, str) else str(context)

        for round_idx in range(max_rounds):
            feedback = await critic.evaluate(query, ctx_str, response)
            logger.info(
                f"Critic round {round_idx + 1}: verdict={feedback.verdict}, "
                f"confidence={feedback.confidence:.2f}, "
                f"actions={feedback.action_summary()}"
            )

            if not feedback.should_continue:
                break

            if critic_mode == "blind":
                # Ablation: ignore structured feedback, re-retrieve with original query
                refined_q = query
            else:
                # Directed mode: use structured feedback to build targeted query
                refined_q = feedback.refined_query if feedback.refined_query else query
                if feedback.has_missing_entities:
                    entity_hint = ", ".join(feedback.missing_entities)
                    refined_q = f"{refined_q} (focus on: {entity_hint})"

            try:
                new_context = await self._retrieve_relevant_contexts(query=refined_q)
                if new_context:
                    context = self._merge_contexts(context, new_context)
                    ctx_str = context if isinstance(context, str) else str(context)
                    if self.config.query_type == "qa":
                        response = await self.generation_qa(query, context)
                    else:
                        response = await self.generation_summary(query, context)
            except Exception as e:
                logger.warning(f"Critic re-retrieval failed at round {round_idx + 1}: {e}", exc_info=True)
                break

        return response, context

    @staticmethod
    def _merge_contexts(old_ctx, new_ctx):
        """Merge two contexts, unifying types to avoid silent data loss."""
        if old_ctx is None:
            return new_ctx
        if new_ctx is None:
            return old_ctx
        # Both str
        if isinstance(old_ctx, str) and isinstance(new_ctx, str):
            return old_ctx + "\n\n--- Additional Retrieved Context ---\n" + new_ctx
        # Both list/tuple
        if isinstance(old_ctx, (list, tuple)) and isinstance(new_ctx, (list, tuple)):
            return list(old_ctx) + list(new_ctx)
        # Mixed types: convert to str and concatenate
        old_str = old_ctx if isinstance(old_ctx, str) else "\n".join(str(x) for x in old_ctx)
        new_str = new_ctx if isinstance(new_ctx, str) else "\n".join(str(x) for x in new_ctx)
        return old_str + "\n\n--- Additional Retrieved Context ---\n" + new_str

    async def _normalize_answer(self, query: str, raw_answer: str) -> str:
        """Apply Answer Normalizer to convert verbose answers to benchmark-aligned form."""
        normalizer = self._get_normalizer()
        try:
            normalized = await normalizer.normalize(query, raw_answer)
            logger.info(f"AnswerNormalizer: '{raw_answer[:80]}...' → '{normalized}'")
            # If normalizer still returns "unknown" but raw answer had content, keep raw answer
            if normalized.strip().lower() == "unknown" and raw_answer.strip().lower() != "unknown":
                logger.info(f"AnswerNormalizer returned 'unknown', falling back to raw answer")
                return raw_answer
            return normalized
        except Exception as e:
            logger.warning(f"AnswerNormalizer failed: {e}, returning raw answer")
            return raw_answer

    @abstractmethod
    async def generation_summary(self, query, context):
        pass

    @abstractmethod
    async def generation_qa(self, query, context):
        pass

    async def extract_query_entities(self, query):
        entities = []
        try:
            ner_messages = GraphPrompt.NER.format(user_input=query)

            response_content = await self.llm.aask(ner_messages)
            entities = prase_json_from_response(response_content)

            if 'named_entities' not in entities:
                entities = []
            else:
                entities = entities['named_entities']

            entities = [clean_str(p) for p in entities]
        except Exception as e:
            logger.error('Error in Retrieval NER: {}'.format(e))

        return entities

    async def extract_query_keywords(self, query, mode="low"):
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        result = await self.llm.aask(kw_prompt)
        keywords = None
        keywords_data = prase_json_from_response(result)
        if mode == "low":
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        elif mode == "high":
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)
        elif mode == "hybrid":
            low_level = keywords_data.get("low_level_keywords", [])
            high_level = keywords_data.get("high_level_keywords", [])
            keywords = [low_level, high_level]

        return keywords

    async def _map_global_communities(
            self,
            query: str,
            communities_data
    ):

        # TODO: support other type of context filter
        community_groups = []
        while len(communities_data):
            this_group = truncate_list_by_token_size(
                communities_data,
                key=lambda x: x["report_string"],
                max_token_size=self.config.global_max_token_for_community_report,
            )
            if not this_group:
                # Single element exceeds max_token_size — take it anyway to avoid infinite loop
                community_groups.append(communities_data[:1])
                communities_data = communities_data[1:]
                continue
            community_groups.append(this_group)
            communities_data = communities_data[len(this_group):]

        async def _process(community_truncated_datas: list[Any]) -> dict:
            communities_section_list = [["id", "content", "rating", "importance"]]
            for i, c in enumerate(community_truncated_datas):
                communities_section_list.append(
                    [
                        i,
                        c["report_string"],
                        c["report_json"].get("rating", 0),
                        c['community_info']['occurrence'],
                    ]
                )
            community_context = list_to_quoted_csv_string(communities_section_list)
            sys_prompt_temp = QueryPrompt.GLOBAL_MAP_RAG_POINTS
            sys_prompt = sys_prompt_temp.format(context_data=community_context)

            response = await self.llm.aask(
                query,
                system_msgs=[sys_prompt]
            )

            data = prase_json_from_response(response)
            return data.get("points", [])

        logger.info(f"Grouping to {len(community_groups)} groups for global search")
        responses = await asyncio.gather(*[_process(c) for c in community_groups])

        return responses
