"""
EvidenceFusion & Candidate Generator.

Takes the (possibly heterogeneous) output from the Retriever Bank — graph
results, text results, or already-hybrid-fused results — and:
  1. Normalizes them into a uniform text evidence string.
  2. Generates an INITIAL candidate answer via a single LLM call.

The candidate answer is then passed to Re-Generation Agent for verification
or refinement.

This is distinct from HybridFusion.py:
  - HybridFusion   : RRF on ranked lists INSIDE the Retriever Bank.
  - EvidenceFusion : prepare evidence + candidate answer for the generator.

Usage:
    ef = EvidenceFusionModule(llm)
    evidence_str, candidate = await ef.fuse_and_generate(question, retriever_output)
"""

from typing import List, Dict, Union, Tuple
from Core.Common.Logger import logger
from Core.Prompt.NewGPrompt import EVIDENCE_FUSION_CANDIDATE_PROMPT


MAX_EVIDENCE_CHARS = 4000


class EvidenceFusionModule:
    """Merge retriever outputs into evidence text + produce candidate answer."""

    def __init__(self, llm, max_chars: int = MAX_EVIDENCE_CHARS):
        self.llm = llm
        self.max_chars = max_chars

    async def fuse_and_generate(
        self,
        question: str,
        retriever_output: Union[str, List, Dict, None],
    ) -> Tuple[str, str]:
        """
        Returns (evidence_str, candidate_answer).
        """
        evidence_str = self.format_evidence(retriever_output)

        if not evidence_str.strip():
            return "", "unknown"

        candidate = await self._generate_candidate(question, evidence_str)
        return evidence_str, candidate

    def format_evidence(self, retriever_output) -> str:
        """Normalize heterogeneous retriever outputs to a single text block."""
        if retriever_output is None:
            return ""

        if isinstance(retriever_output, str):
            return retriever_output[: self.max_chars]

        if isinstance(retriever_output, list):
            lines = []
            total = 0
            for i, item in enumerate(retriever_output):
                snippet = self._render_item(item, i)
                if total + len(snippet) > self.max_chars:
                    break
                lines.append(snippet)
                total += len(snippet)
            return "\n\n".join(lines)

        if isinstance(retriever_output, dict):
            return self._render_item(retriever_output, 0)[: self.max_chars]

        return str(retriever_output)[: self.max_chars]

    @staticmethod
    def _render_item(item, idx: int) -> str:
        if isinstance(item, dict):
            content = (item.get("content")
                       or item.get("description")
                       or item.get("entity_name")
                       or "")
            src = "+".join(item.get("sources", [])) if item.get("sources") else ""
            score = item.get("rrf_score") or item.get("score")
            tag_parts = [f"#{idx + 1}"]
            if src:
                tag_parts.append(src)
            if score is not None:
                tag_parts.append(f"s={float(score):.3f}")
            tag = " ".join(tag_parts)
            return f"[{tag}] {content}"
        return f"[#{idx + 1}] {item}"

    async def _generate_candidate(self, question: str, evidence: str) -> str:
        prompt = EVIDENCE_FUSION_CANDIDATE_PROMPT.format(
            question=question, evidence=evidence
        )
        try:
            response = await self.llm.aask(msg=prompt)
            return self._extract_answer(response)
        except Exception as e:
            logger.warning(f"EvidenceFusion candidate generation failed: {e}")
            return "unknown"

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract the concise candidate answer (strip any preamble)."""
        text = response.strip()
        for marker in ("Candidate Answer:", "candidate answer:", "Answer:", "answer:"):
            if marker in text:
                text = text.split(marker, 1)[-1].strip()
                break
        # First non-empty line only
        for line in text.split("\n"):
            line = line.strip().rstrip(".").strip('"').strip("'")
            if line:
                return line
        return "unknown"
