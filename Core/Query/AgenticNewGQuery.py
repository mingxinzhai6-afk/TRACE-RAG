"""
AgenticNewG: Unified Agentic GraphRAG Query Engine

A single architecture that dynamically routes each query to the best-fit
retriever from a Retriever Bank (HippoRAG / ToG / RAPTOR), with:
  - Innovation 1: Critic-driven re-retrieval (can retry with a DIFFERENT method)
  - Innovation 2: Answer Normalizer
  - Innovation 3: Adaptive Router (query complexity classification)

Unlike per-method Agentic wrappers (AgenticHippoRAG, AgenticToG, ...),
AgenticNewG loads ALL retriever instances at startup and dynamically selects
the best one per query — like AgentG but with an explicit Retriever Bank.

Usage:
    engine = AgenticNewGEngine(instances, router, config)
    answer = await engine.query(question)
"""

from Core.Common.Logger import logger
from Core.Query.QueryRouter import QueryRouter, RouteResult
from Core.Query.CriticModule import CriticModule, CriticFeedback
from Core.Query.AnswerNormalizer import AnswerNormalizer


# Method name → instance key mapping
METHOD_TO_INSTANCE = {
    "ppr": "ppr",
    "tog": "tog",
    "basic": "raptor",
    "raptor": "raptor",
    "hybrid": "hybrid",    # Special: fuses ppr + raptor results
}

# Fallback order: if one method fails Critic, try the next
FALLBACK_ORDER = {
    "ppr": ["tog", "raptor"],
    "tog": ["ppr", "raptor"],
    "raptor": ["ppr", "tog"],
    "hybrid": ["tog", "ppr"],
}


class AgenticNewGEngine:
    """
    Unified query engine that manages multiple GraphRAG instances
    and routes queries dynamically.
    """

    def __init__(self, instances: dict, llm, agentic_config: dict,
                 routing_map: dict = None):
        """
        Args:
            instances: dict of {name: (querier, GraphRAG_instance)} e.g.
                       {"ppr": (PPRQuery, digimon_ppr),
                        "tog": (ToGQuery, digimon_tog),
                        "raptor": (BasicQuery, digimon_raptor)}
            llm: Shared LLM instance (for Router, Critic, Normalizer)
            agentic_config: dict with keys: use_critic, use_answer_normalizer,
                           critic_max_rounds, critic_cross_method_retry, critic_mode
            routing_map: Optional custom routing map override
        """
        self.instances = instances  # {name: querier}
        self.llm = llm
        self.agentic_config = agentic_config

        # Initialize agentic modules
        self.router = QueryRouter(llm, routing_map)
        self.critic = CriticModule(llm) if agentic_config.get("use_critic") else None
        self.normalizer = AnswerNormalizer(llm) if agentic_config.get("use_answer_normalizer") else None

        self.critic_max_rounds = agentic_config.get("critic_max_rounds", 2)
        self.cross_method_retry = agentic_config.get("critic_cross_method_retry", True)
        self.critic_mode = agentic_config.get("critic_mode", "directed")

    async def query(self, question: str) -> dict:
        """
        Full AgenticNewG pipeline for a single question.

        Returns:
            dict with keys: output, route_category, route_confidence,
                           route_method, critic_rounds, methods_tried
        """
        result = {
            "route_category": "",
            "route_confidence": 0.0,
            "route_method": "",
            "critic_rounds": 0,
            "methods_tried": [],
        }

        # === Step 1: Router classifies query ===
        route = await self.router.classify(question)
        result["route_category"] = route.category
        result["route_confidence"] = route.confidence
        result["route_method"] = route.recommended_method

        # === Step 2: Determine which instance to use ===
        instance_key = METHOD_TO_INSTANCE.get(route.recommended_method, "ppr")

        # === Step 3: Query the selected instance ===
        if instance_key == "hybrid":
            response, context = await self._hybrid_query(question)
            result["methods_tried"].append("hybrid(ppr+raptor)")
        else:
            response, context = await self._single_query(instance_key, question)
            result["methods_tried"].append(instance_key)

        # === Step 4: Critic loop with cross-method retry ===
        if self.critic and response:
            response, context, critic_rounds, methods_tried = await self._critic_loop(
                question, context, response, instance_key
            )
            result["critic_rounds"] = critic_rounds
            result["methods_tried"] = methods_tried

        # === Step 5: Answer Normalizer ===
        if self.normalizer and response:
            response = await self._normalize(question, response)

        result["output"] = response
        return result

    async def _single_query(self, instance_key: str, question: str):
        """Query a single instance, return (response, context)."""
        if instance_key not in self.instances:
            logger.warning(f"Instance '{instance_key}' not available, falling back to ppr")
            instance_key = "ppr"

        querier = self.instances[instance_key]
        try:
            # Get context from retriever
            context = await querier._retrieve_relevant_contexts(query=question)
            # Generate answer
            if querier.config.query_type == "qa":
                response = await querier.generation_qa(question, context)
            else:
                response = await querier.generation_summary(question, context)
            return response, context
        except Exception as e:
            logger.error(f"Query failed on instance '{instance_key}': {e}")
            return None, None

    async def _hybrid_query(self, question: str):
        """
        Hybrid mode: query both PPR and RAPTOR, fuse evidence, generate once.
        This implements the "Evidence Fusion" box in the architecture diagram.
        """
        # Query both instances
        resp_ppr, ctx_ppr = await self._single_query("ppr", question)
        resp_raptor, ctx_raptor = await self._single_query("raptor", question)

        # Fuse contexts
        fused_context = self._fuse_contexts(ctx_ppr, ctx_raptor)

        # Generate final answer from fused context using PPR's querier
        querier = self.instances.get("ppr")
        if querier and fused_context:
            try:
                response = await querier.generation_qa(question, fused_context)
                return response, fused_context
            except Exception as e:
                logger.warning(f"Hybrid generation failed: {e}")

        # Fallback: return the better individual response
        if resp_ppr:
            return resp_ppr, ctx_ppr
        return resp_raptor, ctx_raptor

    @staticmethod
    def _fuse_contexts(ctx_a, ctx_b):
        """Fuse two retrieval contexts into one."""
        if ctx_a is None:
            return ctx_b
        if ctx_b is None:
            return ctx_a

        if isinstance(ctx_a, str) and isinstance(ctx_b, str):
            return (ctx_a +
                    "\n\n--- Evidence from Text Retriever ---\n" +
                    ctx_b)
        if isinstance(ctx_a, (list, tuple)) and isinstance(ctx_b, (list, tuple)):
            return list(ctx_a) + list(ctx_b)

        str_a = ctx_a if isinstance(ctx_a, str) else "\n".join(str(x) for x in ctx_a)
        str_b = ctx_b if isinstance(ctx_b, str) else "\n".join(str(x) for x in ctx_b)
        return str_a + "\n\n--- Evidence from Text Retriever ---\n" + str_b

    async def _critic_loop(self, question, context, response, current_method):
        """
        Critic-driven iterative refinement with cross-method retry.

        If the current method's answer fails Critic evaluation AND
        critic_cross_method_retry is enabled, tries a different method
        from the fallback order.
        """
        total_rounds = 0
        methods_tried = [current_method]

        for round_idx in range(self.critic_max_rounds):
            ctx_str = context if isinstance(context, str) else str(context)
            feedback = await self.critic.evaluate(question, ctx_str, response)
            total_rounds += 1

            logger.info(
                f"Critic round {round_idx + 1}: verdict={feedback.verdict}, "
                f"conf={feedback.confidence:.2f}, method={current_method}, "
                f"actions={feedback.action_summary()}"
            )

            if not feedback.should_continue:
                break

            # Decide: same-method re-retrieval or cross-method retry
            retry_different = (
                self.cross_method_retry
                and feedback.confidence < 0.4
                and round_idx < self.critic_max_rounds - 1
            )

            if retry_different:
                # Cross-method retry: pick next untried method
                fallbacks = FALLBACK_ORDER.get(current_method, [])
                next_method = None
                for fb_method in fallbacks:
                    if fb_method not in methods_tried and fb_method in self.instances:
                        next_method = fb_method
                        break

                if next_method:
                    logger.info(f"Critic cross-method retry: {current_method} → {next_method}")
                    new_response, new_context = await self._single_query(
                        next_method, question
                    )
                    if new_response:
                        response = new_response
                        context = new_context
                        current_method = next_method
                        methods_tried.append(next_method)
                        continue

            # Same-method re-retrieval with refined query
            if self.critic_mode == "blind":
                refined_q = question
            else:
                refined_q = feedback.refined_query if feedback.refined_query else question
                if feedback.has_missing_entities:
                    entity_hint = ", ".join(feedback.missing_entities)
                    refined_q = f"{refined_q} (focus on: {entity_hint})"

            # For hybrid, fall back to ppr querier for re-retrieval
            retr_key = current_method if current_method != "hybrid" else "ppr"
            querier = self.instances.get(retr_key)
            if querier:
                try:
                    new_context = await querier._retrieve_relevant_contexts(
                        query=refined_q
                    )
                    if new_context:
                        context = self._fuse_contexts(context, new_context)
                        response = await querier.generation_qa(question, context)
                except Exception as e:
                    logger.warning(f"Critic re-retrieval failed: {e}")
                    break

        return response, context, total_rounds, methods_tried

    async def _normalize(self, question, response):
        """Apply Answer Normalizer."""
        try:
            normalized = await self.normalizer.normalize(question, response)
            logger.info(f"Normalizer: '{response[:60]}...' → '{normalized}'")
            if normalized.strip().lower() == "unknown" and response.strip().lower() != "unknown":
                return response
            return normalized
        except Exception as e:
            logger.warning(f"Normalizer failed: {e}")
            return response
