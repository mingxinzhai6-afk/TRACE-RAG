"""
Critic-Driven Directed Subgraph Expansion (Innovation 1)

Unlike Agent-G which blindly re-retrieves on Critic failure, our Critic
outputs structured feedback that maps to SPECIFIC retrieval actions:
  - missing_entity: "X"  → expand neighbors of X
  - broken_path: "A→?→B" → directed path completion
  - conflict: "C vs D"   → trigger entity disambiguation

The CriticModule evaluates the current answer, and if insufficient,
returns a CriticFeedback object that the query loop uses to perform
targeted re-retrieval.

Usage:
    critic = CriticModule(llm)
    feedback = await critic.evaluate(question, context, answer)
    if feedback.should_continue:
        refined_query = feedback.refined_query
        # ... use refined_query for targeted re-retrieval
"""

from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict
from Core.Common.Logger import logger
from Core.Common.Utils import prase_json_from_response
from arc_fuse_digimon.prompts import CRITIC_EVALUATE_PROMPT

# Global statistics for ablation / coverage analysis
# Tracks how often each feedback type is triggered across all queries
_feedback_stats = defaultdict(int)  # keys: "pass","missing_entity","broken_path","conflict","total_rounds"


def get_feedback_stats() -> dict:
    """Return a copy of accumulated feedback type statistics."""
    return dict(_feedback_stats)


def reset_feedback_stats():
    """Reset statistics (call at the start of each eval run)."""
    _feedback_stats.clear()


@dataclass
class CriticFeedback:
    """Structured feedback from the Critic module."""
    verdict: str = "pass"              # "pass" | "retrieve_more" | "revise"
    confidence: float = 1.0
    missing_entities: List[str] = field(default_factory=list)
    broken_paths: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    suggestion: str = ""
    refined_query: str = ""

    @property
    def should_continue(self) -> bool:
        """Whether the query loop should do another retrieval round."""
        return self.verdict in ("retrieve_more", "revise")

    @property
    def has_missing_entities(self) -> bool:
        return len(self.missing_entities) > 0

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    def action_summary(self) -> str:
        """Human-readable summary of recommended actions."""
        actions = []
        if self.missing_entities:
            actions.append(f"Expand entities: {', '.join(self.missing_entities)}")
        if self.broken_paths:
            actions.append(f"Complete paths: {', '.join(self.broken_paths)}")
        if self.conflicts:
            actions.append(f"Disambiguate: {', '.join(self.conflicts)}")
        if self.suggestion:
            actions.append(f"Suggestion: {self.suggestion}")
        return " | ".join(actions) if actions else "No action needed"


class CriticModule:
    """
    Evaluates answer quality and provides structured feedback for
    directed re-retrieval.
    """

    def __init__(self, llm):
        self.llm = llm

    async def evaluate(
        self,
        question: str,
        context: str,
        answer: str
    ) -> CriticFeedback:
        """
        Evaluate the current answer against the question and context.

        Args:
            question: The original question.
            context: The retrieved context (may be truncated).
            answer: The current generated answer.

        Returns:
            CriticFeedback with structured analysis.
        """
        # Truncate context to avoid token overflow
        ctx_preview = context[:3000] if isinstance(context, str) else str(context)[:3000]

        prompt = CRITIC_EVALUATE_PROMPT.format(
            question=question,
            context=ctx_preview,
            answer=answer
        )

        try:
            response = await self.llm.aask(msg=prompt)
            fb = self._parse_feedback(response)
            # Record statistics for coverage analysis
            _feedback_stats["total_rounds"] += 1
            _feedback_stats[fb.verdict] += 1
            if fb.missing_entities:
                _feedback_stats["missing_entity"] += 1
            if fb.broken_paths:
                _feedback_stats["broken_path"] += 1
            if fb.conflicts:
                _feedback_stats["conflict"] += 1
            return fb
        except Exception as e:
            logger.warning(f"CriticModule evaluation failed: {e}")
            return CriticFeedback(verdict="pass", confidence=0.5)

    @staticmethod
    def _ensure_list(value) -> list:
        """Ensure a value is a list — LLM may return str instead of list."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # "Entity A, Entity B" → ["Entity A", "Entity B"]
            return [v.strip() for v in value.split(",") if v.strip()]
        if value is None:
            return []
        return [value]

    @staticmethod
    def _suggestion_requests_followup(suggestion: str) -> bool:
        """Whether the Critic's suggestion asks for more verification/retrieval."""
        text = str(suggestion or "").strip().lower()
        if not text:
            return False

        negative_markers = (
            "no additional", "no further", "not needed", "nothing additional",
            "current evidence is sufficient", "answer is sufficient",
            "well-supported",
        )
        if any(marker in text for marker in negative_markers):
            return False

        followup_markers = (
            "retrieve", "retrieval", "re-retrieve", "search", "look up",
            "find", "confirm", "verify", "expand", "get more", "more evidence",
            "additional evidence", "specific evidence", "more specific",
            "missing", "complete the path", "complete path",
        )
        return any(marker in text for marker in followup_markers)

    def _guard_premature_pass(self, feedback: CriticFeedback) -> CriticFeedback:
        """Coerce pass verdicts that still contain actionable follow-up feedback."""
        if feedback.verdict != "pass":
            return feedback

        has_retrieval_gap = bool(
            feedback.missing_entities
            or feedback.broken_paths
            or feedback.refined_query.strip()
            or self._suggestion_requests_followup(feedback.suggestion)
        )
        if feedback.conflicts:
            feedback.verdict = "revise"
        elif has_retrieval_gap:
            feedback.verdict = "retrieve_more"

        if feedback.verdict != "pass":
            logger.info(
                f"CriticModule coerced pass verdict to {feedback.verdict} "
                "due to actionable feedback"
            )
        return feedback

    def _parse_feedback(self, response: str) -> CriticFeedback:
        """Parse LLM JSON response into CriticFeedback."""
        try:
            data = prase_json_from_response(response)
            if not isinstance(data, dict):
                return CriticFeedback(verdict="pass", confidence=0.5)

            feedback_data = data.get("feedback", {})
            if not isinstance(feedback_data, dict):
                feedback_data = {}

            verdict = str(data.get("verdict", "pass")).strip().lower()
            if verdict not in ("pass", "retrieve_more", "revise"):
                verdict = "pass"

            feedback = CriticFeedback(
                verdict=verdict,
                confidence=float(data.get("confidence", 0.5)),
                missing_entities=self._ensure_list(feedback_data.get("missing_entities", [])),
                broken_paths=self._ensure_list(feedback_data.get("broken_paths", [])),
                conflicts=self._ensure_list(feedback_data.get("conflicts", [])),
                suggestion=str(feedback_data.get("suggestion", "")),
                refined_query=str(data.get("refined_query", "")),
            )
            return self._guard_premature_pass(feedback)
        except Exception as e:
            logger.warning(f"CriticModule JSON parse failed: {e}")
            return CriticFeedback(verdict="pass", confidence=0.5)
