"""
Query Complexity-Aware Routing (Innovation 3)

Unlike Agent-G which uses a full LLM call for routing (expensive),
this module uses a lightweight few-shot classifier to determine query
complexity and route to the appropriate retrieval strategy.

Query categories → Recommended retrieval methods (DIGIMON framework):
  - factoid (single-hop)  → Entity/Text Retriever (BasicQuery/PPRQuery)
  - multi_hop             → Graph Retriever (HippoRAG/ToG)
  - comparative           → Hybrid Retriever (LightRAG hybrid mode)
  - aggregation           → Community-based (GGraphRAG/LGraphRAG)
  - reasoning             → Graph + CoT (ToG/HippoRAG with IR-CoT)

The DIGIMON framework already has multiple retrieval methods as a natural
Retriever Bank — this module adds the routing layer.

Usage:
    router = QueryRouter(llm)
    route = await router.classify(question)
    method = router.route_to_method(route.category)
"""

from dataclasses import dataclass
from Core.Common.Logger import logger
from Core.Common.Utils import prase_json_from_response
from Core.Prompt import QueryPrompt


# Maps query category → recommended DIGIMON retriever query_type
# This mapping can be customized per dataset via YAML config
DEFAULT_ROUTING_MAP = {
    "factoid": "ppr",       # HippoRAG PPR — efficient for single-hop
    "multi_hop": "tog",     # ToG — iterative reasoning across entities
    "comparative": "basic", # BasicQuery with hybrid mode
    "aggregation": "basic", # BasicQuery with community (GGraphRAG)
    "reasoning": "tog",     # ToG — deep multi-step reasoning
}


@dataclass
class RouteResult:
    """Result of query classification."""
    category: str = "factoid"
    confidence: float = 0.5
    recommended_method: str = "ppr"

    def __str__(self):
        return f"[{self.category}] conf={self.confidence:.2f} → {self.recommended_method}"


class QueryRouter:
    """
    Lightweight query complexity classifier that routes questions
    to appropriate retrieval strategies.
    """

    def __init__(self, llm, routing_map: dict = None):
        """
        Args:
            llm: The LLM instance for classification.
            routing_map: Optional custom category→method mapping.
                         Defaults to DEFAULT_ROUTING_MAP.
        """
        self.llm = llm
        self.routing_map = routing_map or DEFAULT_ROUTING_MAP

    async def classify(self, question: str) -> RouteResult:
        """
        Classify a question's complexity and return routing recommendation.

        Uses a single LLM call with few-shot prompt. For production,
        this could be replaced with a fine-tuned small model or
        rule-based heuristics for zero cost.

        Args:
            question: The input question.

        Returns:
            RouteResult with category, confidence, and recommended method.
        """
        prompt = QueryPrompt.QUERY_ROUTER_PROMPT.format(question=question)

        try:
            response = await self.llm.aask(msg=prompt)
            data = prase_json_from_response(response)

            if not isinstance(data, dict):
                return self._fallback(question)

            category = data.get("category", "factoid")
            confidence = float(data.get("confidence", 0.5))

            # Validate category
            if category not in self.routing_map:
                logger.warning(f"QueryRouter: unknown category '{category}', falling back to factoid")
                category = "factoid"

            recommended = self.routing_map[category]

            result = RouteResult(
                category=category,
                confidence=confidence,
                recommended_method=recommended,
            )
            logger.info(f"QueryRouter: {question[:60]}... → {result}")
            return result

        except Exception as e:
            logger.warning(f"QueryRouter classification failed: {e}")
            return self._fallback(question)

    def _fallback(self, question: str) -> RouteResult:
        """
        Rule-based fallback when LLM classification fails.
        Uses simple keyword heuristics.
        """
        q = question.lower().strip()

        # Multi-hop indicators
        multi_hop_keywords = ["what is the", "who is the", "where is the", "born in the country",
                              "capital of", "president of the country"]
        if any(kw in q for kw in multi_hop_keywords) and len(q.split()) > 10:
            return RouteResult("multi_hop", 0.4, self.routing_map["multi_hop"])

        # Comparative indicators
        if any(kw in q for kw in ["compare", "difference between", "which is", "who is older",
                                   "who is taller", "larger", "smaller", "more"]):
            return RouteResult("comparative", 0.4, self.routing_map["comparative"])

        # Aggregation indicators
        if any(kw in q for kw in ["list all", "how many", "name all", "what are the"]):
            return RouteResult("aggregation", 0.4, self.routing_map["aggregation"])

        # Default: factoid
        return RouteResult("factoid", 0.3, self.routing_map["factoid"])

    def route_to_method(self, category: str) -> str:
        """Get the recommended DIGIMON query_type for a category."""
        return self.routing_map.get(category, "ppr")
