"""
ARC-Fuse multi-agent judge framework for answer generation.

Architecture (JAILJUDGE Multi-agent Judge Framework adapted for RAG):

  [Judging Agents × N]      Each independently scores evidence + gives candidate answer
         ↓
  [Evidence Aggregation]    μ_avg formula aggregates scores, merges reasons
         ↓
  [Voting Agents × N]       Each sees all judge outputs, votes on best answer
         ↓
  [Inference Agent]         Final CoT reasoning → definitive answer

On Commendor "poor_generation" verdict, re_generate_with_feedback() skips
judge/vote and goes straight to the Inference Agent with corrective feedback.

Usage:
    agent = ReGenerationAgent(llm, n_judges=3, n_voters=3)
    answer = await agent.generate(question, evidence, candidate)
    answer = await agent.re_generate_with_feedback(
        question, evidence, prev_answer, feedback
    )
"""

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

from Core.Common.Logger import logger
from Core.Common.Utils import prase_json_from_response
from arc_fuse_digimon.prompts import (
    JUDGE_PROMPT,
    VOTING_PROMPT,
    INFERENCE_COT_PROMPT,
    REGENERATION_RETRY_PROMPT,
)


@dataclass
class JudgeResult:
    score: float
    reason: str
    candidate: str


@dataclass
class VoteResult:
    vote: str
    reason: str


class ReGenerationAgent:
    """Multi-agent judge framework for answer (re-)generation."""

    def __init__(self, llm, n_judges: int = 3, n_voters: int = 3):
        self.llm = llm
        self.n_judges = n_judges
        self.n_voters = n_voters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        question: str,
        evidence: str,
        candidate: str = "",
        feedback: str = "",
    ) -> str:
        """Full multi-agent pipeline: Judge → Aggregate → Vote → Infer.

        feedback: if non-empty (e.g. from Commendor poor_generation), passed to
        the Inference Agent so it knows what went wrong in the previous attempt.
        After this call, self.last_debug holds per-round diagnostics.
        """
        # Step 1: Judging Agents (parallel)
        judge_results = await self._judge_phase(question, evidence)

        # Step 2: Evidence Aggregation (μ_avg)
        agg_score, merged_reason, best_candidate = self._aggregate(
            judge_results, candidate
        )

        # Step 3: Voting Agents (parallel, each sees all judge outputs)
        vote_results = await self._voting_phase(
            question, evidence, judge_results, agg_score, merged_reason
        )

        # Step 4: Inference Agent (best_candidate as fallback if votes all fail)
        final_answer = await self._inference_phase(
            question, evidence, agg_score, merged_reason, vote_results,
            fallback_answer=best_candidate,
            feedback=feedback,
        )

        # Expose diagnostics for callers that want per-round detail
        self.last_debug = {
            "judge_scores": [round(r.score, 2) for r in judge_results],
            "judge_candidates": [r.candidate for r in judge_results],
            "judge_reasons": [r.reason for r in judge_results],
            "agg_score": round(agg_score, 2),
            "best_candidate": best_candidate,
            "votes": [v if isinstance(v, str) else getattr(v, "vote", str(v))
                      for v in vote_results],
            "final_answer": final_answer,
        }
        return final_answer

    async def re_generate_with_feedback(
        self,
        question: str,
        evidence: str,
        prev_answer: str,
        feedback: str,
    ) -> str:
        """Skip judge/vote — use Commendor hint to directly re-infer."""
        prompt = REGENERATION_RETRY_PROMPT.format(
            question=question,
            evidence=evidence or "none",
            prev_answer=prev_answer,
            feedback=feedback or "The previous answer was incorrect or incomplete.",
        )
        try:
            response = await self.llm.aask(msg=prompt)
            return self._extract_final_answer(response)
        except Exception as e:
            logger.warning(f"ReGenerationAgent.re_generate_with_feedback failed: {e}")
            return prev_answer or "unknown"

    # ------------------------------------------------------------------
    # Step 1: Judging Agents
    # ------------------------------------------------------------------

    async def _judge_phase(self, question: str, evidence: str) -> List[JudgeResult]:
        """Run N judging agents in parallel."""
        tasks = [
            self._single_judge(question, evidence, judge_id=i)
            for i in range(self.n_judges)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, JudgeResult)]
        if not valid:
            valid = [JudgeResult(score=5.0, reason="evaluation unavailable", candidate="unknown")]
        logger.info(
            f"JudgePhase: {len(valid)} results, "
            f"scores={[round(r.score, 1) for r in valid]}, "
            f"candidates={[r.candidate for r in valid]}"
        )
        return valid

    async def _single_judge(
        self, question: str, evidence: str, judge_id: int
    ) -> JudgeResult:
        prompt = JUDGE_PROMPT.format(
            judge_id=judge_id + 1,
            question=question,
            evidence=(evidence or "none")[:3000],
        )
        try:
            response = await self.llm.aask(msg=prompt)
            data = prase_json_from_response(response)
            if not isinstance(data, dict):
                raise ValueError("non-dict response")
            return JudgeResult(
                score=max(0.0, min(10.0, float(data.get("score", 5.0)))),
                reason=str(data.get("reason", ""))[:300],
                candidate=str(data.get("candidate", "unknown")),
            )
        except Exception as e:
            logger.warning(f"Judge {judge_id + 1} failed: {e}")
            return JudgeResult(score=5.0, reason="evaluation failed", candidate="unknown")

    # ------------------------------------------------------------------
    # Step 2: Evidence Aggregation  μ_avg = (1/M) Σ score_i
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        judge_results: List[JudgeResult],
        prior_candidate: str = "",
    ) -> Tuple[float, str, str]:
        """
        Aggregate judge scores with μ_avg formula.
        Returns (aggregated_score, merged_reason, best_candidate).
        best_candidate is picked by majority vote among judge candidates.
        """
        if not judge_results:
            return 5.0, "no evaluation", prior_candidate or "unknown"

        agg_score = sum(r.score for r in judge_results) / len(judge_results)

        reasons = [
            r.reason for r in judge_results
            if r.reason and r.reason not in ("evaluation failed", "evaluation unavailable")
        ]
        merged_reason = " | ".join(reasons) if reasons else "no evaluation available"

        candidates = [
            r.candidate for r in judge_results
            if r.candidate and r.candidate.lower() != "unknown"
        ]
        if candidates:
            best_candidate = Counter(candidates).most_common(1)[0][0]
        else:
            best_candidate = prior_candidate or "unknown"

        logger.info(
            f"EvidenceAggregation: μ_avg={agg_score:.2f}/10, "
            f"best_candidate='{best_candidate}'"
        )
        return agg_score, merged_reason, best_candidate

    # ------------------------------------------------------------------
    # Step 3: Voting Agents
    # ------------------------------------------------------------------

    async def _voting_phase(
        self,
        question: str,
        evidence: str,
        judge_results: List[JudgeResult],
        agg_score: float,
        merged_reason: str,
    ) -> List[VoteResult]:
        """Run N voting agents in parallel; each sees ALL judge outputs (communication)."""
        judge_summary = self._format_judge_outputs(judge_results)
        tasks = [
            self._single_voter(
                question, evidence, judge_summary, agg_score, merged_reason, voter_id=i
            )
            for i in range(self.n_voters)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, VoteResult)]
        logger.info(
            f"VotingPhase: {len(valid)} votes, "
            f"votes={[r.vote for r in valid]}"
        )
        return valid

    async def _single_voter(
        self,
        question: str,
        evidence: str,
        judge_summary: str,
        agg_score: float,
        merged_reason: str,
        voter_id: int,
    ) -> VoteResult:
        prompt = VOTING_PROMPT.format(
            voter_id=voter_id + 1,
            question=question,
            evidence=(evidence or "none")[:2000],
            judge_outputs=judge_summary,
            aggregated_score=agg_score,
            merged_reason=merged_reason,
        )
        try:
            response = await self.llm.aask(msg=prompt)
            data = prase_json_from_response(response)
            if not isinstance(data, dict):
                raise ValueError("non-dict response")
            return VoteResult(
                vote=str(data.get("vote", "unknown")),
                reason=str(data.get("reason", ""))[:200],
            )
        except Exception as e:
            logger.warning(f"Voter {voter_id + 1} failed: {e}")
            return VoteResult(vote="unknown", reason="voting failed")

    # ------------------------------------------------------------------
    # Step 4: Inference Agent
    # ------------------------------------------------------------------

    async def _inference_phase(
        self,
        question: str,
        evidence: str,
        agg_score: float,
        merged_reason: str,
        vote_results: List[VoteResult],
        fallback_answer: str = "",
        feedback: str = "",
    ) -> str:
        vote_summary = self._format_vote_outputs(vote_results)
        winning_answer = self._majority_vote(vote_results) or fallback_answer or "unknown"

        prompt = INFERENCE_COT_PROMPT.format(
            question=question,
            evidence=(evidence or "none")[:3000],
            aggregated_score=agg_score,
            merged_reason=merged_reason,
            vote_summary=vote_summary,
            winning_answer=winning_answer,
        )
        if feedback:
            prompt += f"\n\n### Previous attempt was rejected. Feedback:\n{feedback}"
        try:
            response = await self.llm.aask(msg=prompt)
            return self._extract_final_answer(response)
        except Exception as e:
            logger.warning(f"InferenceAgent failed: {e}")
            return winning_answer or fallback_answer or "unknown"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_judge_outputs(judge_results: List[JudgeResult]) -> str:
        lines = []
        for i, jr in enumerate(judge_results):
            lines.append(
                f"Judge {i+1}: score={jr.score:.1f}/10, "
                f"candidate=\"{jr.candidate}\", reason=\"{jr.reason}\""
            )
        return "\n".join(lines)

    @staticmethod
    def _format_vote_outputs(vote_results: List[VoteResult]) -> str:
        lines = []
        for i, vr in enumerate(vote_results):
            lines.append(f"Voter {i+1}: vote=\"{vr.vote}\", reason=\"{vr.reason}\"")
        return "\n".join(lines)

    @staticmethod
    def _majority_vote(vote_results: List[VoteResult]) -> str:
        votes = [
            vr.vote for vr in vote_results
            if vr.vote and vr.vote.lower() != "unknown"
        ]
        if not votes:
            return "unknown"
        return Counter(votes).most_common(1)[0][0]

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        """Extract the concise final answer from a CoT response."""
        text = response.strip()
        for marker in (
            "Final Answer:", "final answer:",
            "Final answer:", "FINAL ANSWER:",
            "Answer:", "answer:",
        ):
            if marker in text:
                ans = text.split(marker, 1)[-1].strip()
                for line in ans.split("\n"):
                    line = line.strip().rstrip(".").strip('"').strip("'")
                    if line:
                        return line
        for line in reversed(text.split("\n")):
            line = line.strip().rstrip(".").strip('"').strip("'")
            if line:
                return line
        return "unknown"
