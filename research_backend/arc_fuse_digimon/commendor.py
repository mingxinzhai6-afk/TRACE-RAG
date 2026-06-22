"""
Commendor — 3-way decision module that consumes the Critic Module output
and routes the pipeline to the correct recovery path.

Unlike Agent-G's binary Validator + ICL Commentor, the Commendor combines
validation and diagnosis into ONE LLM call that outputs a categorical
decision ∈ {pass, wrong_retriever, insufficient_evidence, poor_generation}.

Downstream routing:
  - pass                   → continue to Answer Normalizer
  - wrong_retriever        → back to Adaptive Router (switch retriever)
  - insufficient_evidence  → back to Retriever Bank (re-retrieve with same method)
  - poor_generation        → back to Re-Generation Agent (re-generate same evidence)

Usage:
    commendor = Commendor(llm)
    decision = await commendor.diagnose(
        question, action, evidence, answer, critic_feedback
    )
    if decision.is_pass:
        ...
    elif decision.kind == "wrong_retriever":
        ...
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
from Core.Common.Logger import logger
from Core.Common.Utils import prase_json_from_response
from arc_fuse_digimon.prompts import COMMENDOR_PROMPT


# Global stats for ablation analysis
_commendor_stats = defaultdict(int)


def get_commendor_stats() -> dict:
    return dict(_commendor_stats)


def reset_commendor_stats():
    _commendor_stats.clear()


VALID_DECISIONS = {"pass", "wrong_retriever", "insufficient_evidence", "poor_generation"}


@dataclass
class CommendorDecision:
    """3-way decision from the Commendor."""
    kind: str = "pass"             # pass | wrong_retriever | insufficient_evidence | poor_generation
    confidence: float = 1.0
    reason: str = ""
    hint: str = ""

    @property
    def is_pass(self) -> bool:
        return self.kind == "pass"

    @property
    def needs_retriever_switch(self) -> bool:
        return self.kind == "wrong_retriever"

    @property
    def needs_more_evidence(self) -> bool:
        return self.kind == "insufficient_evidence"

    @property
    def needs_regeneration(self) -> bool:
        return self.kind == "poor_generation"


class Commendor:
    """3-way diagnostic decision module."""

    def __init__(self, llm):
        self.llm = llm

    async def diagnose(
        self,
        question: str,
        selection: str,
        entities: str,
        relations: str,
        evidence: str,
        answer: str,
        critic_feedback: str = "",
    ) -> CommendorDecision:
        """
        Categorize the cause of a failed answer.

        Args:
            question:       original question.
            selection:      retriever selection that was used (graph/text/hybrid).
            entities:       comma-separated topic entities string.
            relations:      comma-separated useful relations string.
            evidence:       retrieved evidence (truncated to 2000 chars).
            answer:         generated answer that triggered the diagnosis.
            critic_feedback: prior Critic module output (optional).
        """
        prompt = COMMENDOR_PROMPT.format(
            question=question,
            selection=selection,
            entities=entities or "none",
            relations=relations or "none",
            evidence=(evidence or "")[:2000],
            answer=answer,
            critic_feedback=critic_feedback or "not provided",
        )

        try:
            response = await self.llm.aask(msg=prompt)
            decision = self._parse(response)
        except Exception as e:
            logger.warning(f"Commendor diagnose failed: {e}")
            decision = CommendorDecision(kind="pass", confidence=0.5, reason="LLM error")

        _commendor_stats["total"] += 1
        _commendor_stats[decision.kind] += 1
        logger.info(
            f"Commendor: kind={decision.kind}, conf={decision.confidence:.2f}, "
            f"hint={decision.hint[:120]}"
        )
        return decision

    @staticmethod
    def _parse(response: str) -> CommendorDecision:
        try:
            data = prase_json_from_response(response)
            if not isinstance(data, dict):
                return CommendorDecision(kind="pass", confidence=0.5)

            kind = str(data.get("decision", "pass")).strip().lower()
            if kind not in VALID_DECISIONS:
                kind = "pass"

            return CommendorDecision(
                kind=kind,
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", ""))[:500],
                hint=str(data.get("hint", ""))[:300],
            )
        except Exception as e:
            logger.warning(f"Commendor JSON parse failed: {e}")
            return CommendorDecision(kind="pass", confidence=0.5)
