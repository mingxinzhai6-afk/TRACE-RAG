from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import prompts
from .disambiguation import deduplicate_entities
from .fusion import normalize_results, rrf_fuse
from .interfaces import AsyncLLM, AsyncRetriever
from .json_utils import ensure_string_list, parse_json_object
from .models import (
    CommendorDecision,
    CriticFeedback,
    PipelineConfig,
    PipelineResult,
    QueryAction,
)


def _extract_short_answer(response: str) -> str:
    text = (response or "").strip()
    for marker in (
        "Final Answer:",
        "final answer:",
        "Candidate Answer:",
        "candidate answer:",
        "Normalized Answer:",
        "Answer:",
    ):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    for line in text.splitlines():
        value = line.strip().strip("\"'").rstrip(".")
        if value:
            return value
    return "unknown"


class QueryUnderstanding:
    VALID_SELECTIONS = {"graph", "text", "hybrid"}

    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    async def extract(
        self,
        question: str,
        previous: QueryAction | None = None,
        feedback: str = "",
    ) -> QueryAction:
        if previous and feedback:
            prompt = prompts.ROUTE_REFLECT_PROMPT.format(
                question=question,
                previous_action=previous.to_dict(),
                feedback=feedback,
            )
        else:
            prompt = prompts.ROUTE_PROMPT.format(question=question)

        try:
            response = await self.llm.complete(prompt)
            data = parse_json_object(response)
        except Exception:
            return QueryAction(
                selection=self._heuristic_selection(question),
                route_source="heuristic",
            )

        entities = [
            re.sub(r"\s*\([^)]*\)\s*$", "", entity).strip()
            for entity in ensure_string_list(
                data.get("entities", data.get("topic_entities"))
            )
        ]
        entities = [entity for entity in entities if entity]
        relations = ensure_string_list(
            data.get("relations", data.get("useful_relations"))
        )
        selection = str(data.get("selection", "")).strip().lower()
        source = "llm"
        if selection not in self.VALID_SELECTIONS:
            selection = self._heuristic_selection(question)
            source = "heuristic"

        return QueryAction(
            domain=str(data.get("domain", "general")).strip().lower() or "general",
            entities=entities,
            relations=relations,
            selection=selection,
            route_source=source,
            raw=response,
        )

    @staticmethod
    def _heuristic_selection(question: str) -> str:
        lower = question.lower()
        if any(
            marker in lower
            for marker in (
                "compare",
                "same state",
                "same country",
                "where the",
                "country where",
                "named after",
            )
        ):
            return "hybrid"
        if any(
            marker in lower
            for marker in (
                "explain",
                "describe",
                "why",
                "how does",
                "history of",
                "tell me about",
            )
        ):
            return "text"
        return "graph"


class EvidenceFusion:
    def __init__(self, llm: AsyncLLM, max_characters: int = 5000):
        self.llm = llm
        self.max_characters = max_characters

    def format(self, results: Sequence[Mapping[str, Any]]) -> str:
        blocks: list[str] = []
        total = 0
        for index, result in enumerate(results):
            source = "+".join(result.get("sources", []))
            score = result.get("rrf_score", result.get("score"))
            metadata = [f"#{index + 1}"]
            if source:
                metadata.append(source)
            if score is not None:
                metadata.append(f"score={float(score):.4f}")
            block = f"[{' '.join(metadata)}] {result.get('content', '')}".strip()
            if total + len(block) > self.max_characters:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks)

    async def candidate(self, question: str, evidence: str) -> str:
        if not evidence:
            return "unknown"
        response = await self.llm.complete(
            prompts.CANDIDATE_PROMPT.format(
                question=question,
                evidence=evidence,
            )
        )
        return _extract_short_answer(response)


@dataclass
class JudgeResult:
    score: float
    reason: str
    candidate: str


@dataclass
class VoteResult:
    vote: str
    reason: str


class RegenerationAgent:
    def __init__(self, llm: AsyncLLM, n_judges: int, n_voters: int):
        self.llm = llm
        self.n_judges = n_judges
        self.n_voters = n_voters
        self.last_debug: dict[str, Any] = {}

    async def generate(
        self,
        question: str,
        evidence: str,
        fallback: str = "unknown",
    ) -> str:
        judges = await asyncio.gather(
            *[
                self._judge(index + 1, question, evidence)
                for index in range(self.n_judges)
            ]
        )
        score = sum(judge.score for judge in judges) / max(1, len(judges))
        candidates = [
            judge.candidate
            for judge in judges
            if judge.candidate and judge.candidate.lower() != "unknown"
        ]
        winning = Counter(candidates).most_common(1)[0][0] if candidates else fallback
        judge_outputs = "\n".join(
            f"Judge {index + 1}: score={judge.score}, "
            f"candidate={judge.candidate}, reason={judge.reason}"
            for index, judge in enumerate(judges)
        )
        voters = await asyncio.gather(
            *[
                self._vote(
                    index + 1,
                    question,
                    evidence,
                    judge_outputs,
                )
                for index in range(self.n_voters)
            ]
        )
        votes = [
            voter.vote
            for voter in voters
            if voter.vote and voter.vote.lower() != "unknown"
        ]
        if votes:
            winning = Counter(votes).most_common(1)[0][0]

        response = await self.llm.complete(
            prompts.INFERENCE_PROMPT.format(
                question=question,
                evidence=evidence,
                aggregated_score=f"{score:.2f}",
                votes=", ".join(votes) or "none",
                winning_answer=winning,
            )
        )
        answer = _extract_short_answer(response)
        self.last_debug = {
            "judge_scores": [judge.score for judge in judges],
            "judge_candidates": [judge.candidate for judge in judges],
            "aggregated_score": score,
            "votes": votes,
            "winning_answer": winning,
        }
        return answer if answer.lower() != "unknown" else winning

    async def revise(
        self,
        question: str,
        evidence: str,
        previous_answer: str,
        feedback: str,
    ) -> str:
        response = await self.llm.complete(
            prompts.REVISE_PROMPT.format(
                question=question,
                evidence=evidence,
                previous_answer=previous_answer,
                feedback=feedback,
            )
        )
        return _extract_short_answer(response)

    async def _judge(
        self,
        judge_id: int,
        question: str,
        evidence: str,
    ) -> JudgeResult:
        try:
            response = await self.llm.complete(
                prompts.JUDGE_PROMPT.format(
                    judge_id=judge_id,
                    question=question,
                    evidence=evidence[:4000],
                )
            )
            data = parse_json_object(response)
            return JudgeResult(
                score=max(0.0, min(10.0, float(data.get("score", 0.0)))),
                reason=str(data.get("reason", ""))[:300],
                candidate=str(data.get("candidate", "unknown")).strip(),
            )
        except Exception:
            return JudgeResult(0.0, "judge failed", "unknown")

    async def _vote(
        self,
        voter_id: int,
        question: str,
        evidence: str,
        judge_outputs: str,
    ) -> VoteResult:
        try:
            response = await self.llm.complete(
                prompts.VOTE_PROMPT.format(
                    voter_id=voter_id,
                    question=question,
                    evidence=evidence[:3000],
                    judge_outputs=judge_outputs,
                )
            )
            data = parse_json_object(response)
            return VoteResult(
                vote=str(data.get("vote", "unknown")).strip(),
                reason=str(data.get("reason", ""))[:300],
            )
        except Exception:
            return VoteResult("unknown", "voter failed")


class Critic:
    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    async def evaluate(
        self,
        question: str,
        evidence: str,
        answer: str,
    ) -> CriticFeedback:
        try:
            response = await self.llm.complete(
                prompts.CRITIC_PROMPT.format(
                    question=question,
                    evidence=evidence[:4000],
                    answer=answer,
                )
            )
            data = parse_json_object(response)
            feedback_data = data.get("feedback", {})
            if not isinstance(feedback_data, dict):
                feedback_data = {}
            verdict = str(data.get("verdict", "pass")).strip().lower()
            if verdict not in {"pass", "retrieve_more", "revise"}:
                verdict = "retrieve_more"
            feedback = CriticFeedback(
                verdict=verdict,
                confidence=float(data.get("confidence", 0.5)),
                missing_entities=ensure_string_list(
                    feedback_data.get("missing_entities")
                ),
                broken_paths=ensure_string_list(feedback_data.get("broken_paths")),
                conflicts=ensure_string_list(feedback_data.get("conflicts")),
                suggestion=str(feedback_data.get("suggestion", "")).strip(),
                refined_query=str(data.get("refined_query", "")).strip(),
            )
            return self._guard_actionable_pass(feedback)
        except Exception:
            return CriticFeedback(
                verdict="retrieve_more",
                confidence=0.0,
                suggestion="Critic parsing failed; retry with broader evidence.",
            )

    @staticmethod
    def _guard_actionable_pass(feedback: CriticFeedback) -> CriticFeedback:
        if feedback.verdict != "pass":
            return feedback
        actionable = bool(
            feedback.missing_entities
            or feedback.broken_paths
            or feedback.conflicts
            or feedback.refined_query
        )
        suggestion = feedback.suggestion.lower()
        if any(
            marker in suggestion
            for marker in ("retrieve", "search", "verify", "expand", "missing")
        ):
            actionable = True
        if actionable:
            feedback.verdict = "revise" if feedback.conflicts else "retrieve_more"
        return feedback


class Commendor:
    VALID = {
        "pass",
        "wrong_retriever",
        "insufficient_evidence",
        "poor_generation",
    }

    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    async def diagnose(
        self,
        question: str,
        action: QueryAction,
        evidence: str,
        answer: str,
        critic_feedback: CriticFeedback,
    ) -> CommendorDecision:
        try:
            response = await self.llm.complete(
                prompts.COMMENDOR_PROMPT.format(
                    question=question,
                    selection=action.selection,
                    evidence=evidence[:2500],
                    answer=answer,
                    critic_feedback=critic_feedback.action_summary(),
                )
            )
            data = parse_json_object(response)
            kind = str(data.get("decision", "pass")).strip().lower()
            if kind not in self.VALID:
                kind = "pass"
            return CommendorDecision(
                kind=kind,
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", ""))[:500],
                hint=str(data.get("hint", ""))[:300],
            )
        except Exception:
            return CommendorDecision(
                kind="insufficient_evidence",
                confidence=0.0,
                reason="Commendor parsing failed",
                hint="broaden retrieval",
            )


class AnswerNormalizer:
    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    async def normalize(self, question: str, answer: str) -> str:
        extracted = self._rule_extract(answer)
        try:
            response = await self.llm.complete(
                prompts.NORMALIZE_PROMPT.format(
                    question=question,
                    answer=extracted,
                )
            )
            normalized = _extract_short_answer(response)
            return normalized if normalized else extracted
        except Exception:
            return extracted

    @staticmethod
    def _rule_extract(answer: str) -> str:
        text = (answer or "").strip()
        match = re.search(
            r"(?:final\s+answer|answer)\s*:\s*(.+)",
            text,
            re.IGNORECASE,
        )
        if match:
            text = match.group(1).strip()
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return first.strip("\"'").rstrip(".") or "unknown"


class ArcFusePipeline:
    """Core ARC-Fuse graph-text retrieval and Critic iteration pipeline."""

    def __init__(
        self,
        *,
        llm: AsyncLLM,
        graph_retriever: AsyncRetriever,
        text_retriever: AsyncRetriever,
        config: PipelineConfig | None = None,
    ):
        self.llm = llm
        self.graph_retriever = graph_retriever
        self.text_retriever = text_retriever
        self.config = config or PipelineConfig()
        self.query_understanding = QueryUnderstanding(llm)
        self.evidence_fusion = EvidenceFusion(llm)
        self.regeneration = RegenerationAgent(
            llm,
            n_judges=self.config.n_judges,
            n_voters=self.config.n_voters,
        )
        self.critic = Critic(llm)
        self.commendor = Commendor(llm)
        self.normalizer = AnswerNormalizer(llm)

    async def query(self, question: str) -> PipelineResult:
        if self.config.use_routing:
            action = await self.query_understanding.extract(question)
        else:
            action = QueryAction(
                selection="hybrid",
                route_source="routing_disabled",
            )
        initial_action = QueryAction(**action.to_dict())
        selection_history: list[str] = []
        round_details: list[dict[str, Any]] = []
        retrieval_query = question
        answer = "unknown"

        for round_index in range(self.config.max_rounds):
            selection_history.append(action.selection)
            results = await self._retrieve(retrieval_query, action)
            evidence = self.evidence_fusion.format(results)
            candidate = await self.evidence_fusion.candidate(question, evidence)
            if self.config.use_regeneration:
                answer = await self.regeneration.generate(
                    question,
                    evidence,
                    fallback=candidate,
                )
            else:
                answer = candidate

            feedback = CriticFeedback()
            if self.config.use_critic:
                feedback = await self.critic.evaluate(question, evidence, answer)

            aggregated_score = float(
                self.regeneration.last_debug.get("aggregated_score", 0.0)
            )
            low_confidence = (
                self.config.use_regeneration
                and aggregated_score < self.config.judge_score_threshold
            )
            detail: dict[str, Any] = {
                "round": round_index + 1,
                "selection": action.selection,
                "route_source": action.route_source,
                "retrieval_query": retrieval_query,
                "retrieved_ids": [result.get("id") for result in results],
                "candidate": candidate,
                "answer": answer,
                "aggregated_score": aggregated_score,
                "critic_verdict": feedback.verdict,
                "critic_confidence": feedback.confidence,
                "critic_actions": feedback.action_summary(),
                "outcome": "",
            }

            if not feedback.should_continue:
                detail["outcome"] = "critic_pass"
                round_details.append(detail)
                break

            if feedback.verdict == "revise" and self.config.use_regeneration:
                revised = await self.regeneration.revise(
                    question,
                    evidence,
                    answer,
                    feedback.action_summary(),
                )
                revised_feedback = await self.critic.evaluate(
                    question,
                    evidence,
                    revised,
                )
                detail["revised_answer"] = revised
                detail["revised_critic_verdict"] = revised_feedback.verdict
                answer = revised
                feedback = revised_feedback
                if not feedback.should_continue:
                    detail["outcome"] = "revision_pass"
                    round_details.append(detail)
                    break

            decision: CommendorDecision | None = None
            if (
                self.config.use_commendor
                and (low_confidence or feedback.should_continue)
            ):
                decision = await self.commendor.diagnose(
                    question,
                    action,
                    evidence,
                    answer,
                    feedback,
                )
                detail["commendor_decision"] = decision.kind
                detail["commendor_hint"] = decision.hint
                if decision.kind == "pass":
                    detail["outcome"] = "commendor_pass"
                    round_details.append(detail)
                    break
                if (
                    decision.kind == "poor_generation"
                    and self.config.use_regeneration
                ):
                    answer = await self.regeneration.revise(
                        question,
                        evidence,
                        answer,
                        decision.hint or decision.reason,
                    )
                    retry_feedback = await self.critic.evaluate(
                        question,
                        evidence,
                        answer,
                    )
                    detail["commendor_revised_answer"] = answer
                    if not retry_feedback.should_continue:
                        detail["outcome"] = "commendor_revision_pass"
                        round_details.append(detail)
                        break
                    feedback = retry_feedback

            if round_index + 1 >= self.config.max_rounds:
                detail["outcome"] = "max_rounds"
                round_details.append(detail)
                break

            action = await self._next_action(
                question,
                action,
                feedback,
                decision,
                selection_history,
            )
            retrieval_query = self._next_query(
                question,
                feedback,
                decision,
            )
            detail["next_selection"] = action.selection
            detail["next_retrieval_query"] = retrieval_query
            detail["outcome"] = "continue"
            round_details.append(detail)

        if self.config.use_normalizer:
            answer = await self.normalizer.normalize(question, answer)

        return PipelineResult(
            question=question,
            answer=answer,
            initial_action=initial_action,
            selection_history=selection_history,
            rounds=len(round_details),
            round_details=round_details,
        )

    async def _retrieve(
        self,
        query: str,
        action: QueryAction,
    ) -> list[dict[str, Any]]:
        if action.selection == "graph":
            raw = await self.graph_retriever.retrieve(
                query,
                action,
                self.config.top_k,
            )
            graph = normalize_results(raw, "graph")
            return (
                deduplicate_entities(graph)
                if self.config.use_disambiguation
                else graph
            )
        if action.selection == "text":
            raw = await self.text_retriever.retrieve(
                query,
                action,
                self.config.top_k,
            )
            return normalize_results(raw, "text")

        graph_task = self.graph_retriever.retrieve(
            query,
            action,
            self.config.top_k,
        )
        text_task = self.text_retriever.retrieve(
            query,
            action,
            self.config.top_k,
        )
        graph_raw, text_raw = await asyncio.gather(
            graph_task,
            text_task,
            return_exceptions=True,
        )
        graph = (
            []
            if isinstance(graph_raw, Exception)
            else normalize_results(graph_raw, "graph")
        )
        if self.config.use_disambiguation:
            graph = deduplicate_entities(graph)
        text = [] if isinstance(text_raw, Exception) else normalize_results(
            text_raw,
            "text",
        )
        return rrf_fuse(
            graph,
            text,
            top_k=self.config.top_k,
            rrf_k=self.config.rrf_k,
            graph_weight=self.config.graph_weight,
            text_weight=self.config.text_weight,
        )

    async def _next_action(
        self,
        question: str,
        action: QueryAction,
        feedback: CriticFeedback,
        decision: CommendorDecision | None,
        history: list[str],
    ) -> QueryAction:
        next_action = QueryAction(**action.to_dict())
        if decision and decision.kind == "wrong_retriever":
            next_action = await self.query_understanding.extract(
                question,
                previous=action,
                feedback=decision.hint or decision.reason,
            )
            if next_action.selection == history[-1]:
                next_action.selection = self._switch_selection(history[-1])
                next_action.route_source = "forced_switch"
            return next_action

        if feedback.broken_paths and next_action.selection == "text":
            next_action.selection = "hybrid"
            next_action.route_source = "critic_override"
        elif feedback.conflicts and next_action.selection == "graph":
            next_action.selection = "hybrid"
            next_action.route_source = "critic_override"
        elif (
            decision
            and decision.kind == "insufficient_evidence"
            and len(history) >= 2
            and history[-1] == history[-2]
        ):
            next_action.selection = "hybrid"
            next_action.route_source = "evidence_escalation"
        return next_action

    @staticmethod
    def _next_query(
        question: str,
        feedback: CriticFeedback,
        decision: CommendorDecision | None,
    ) -> str:
        if feedback.refined_query:
            return feedback.refined_query
        parts = [question]
        if feedback.missing_entities:
            parts.append("Missing entities: " + ", ".join(feedback.missing_entities))
        if feedback.broken_paths:
            parts.append("Complete paths: " + "; ".join(feedback.broken_paths))
        if feedback.conflicts:
            parts.append("Resolve conflicts: " + "; ".join(feedback.conflicts))
        if feedback.suggestion:
            parts.append(feedback.suggestion)
        if decision and decision.hint and decision.kind != "poor_generation":
            parts.append(decision.hint)
        return " ".join(parts)

    @staticmethod
    def _switch_selection(selection: str) -> str:
        return {
            "graph": "text",
            "text": "hybrid",
            "hybrid": "graph",
        }.get(selection, "hybrid")
