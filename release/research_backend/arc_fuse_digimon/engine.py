"""
ARC-Fuse engine aligned with the paper experiment pipeline.

Per query:

    Query
      -> QueryUnderstanding
      -> Retriever Bank (graph | text | hybrid)
      -> Evidence Fusion
      -> Re-Generation Agent
      -> Critic
      -> (optional) Commendor
      -> Answer Normalizer

Critic is the primary driver of iterative retrieval. Commendor is only used as
an auxiliary diagnosis module when judge confidence is low or when a retriever
switch is needed.
"""

import asyncio
import re
import unicodedata
from collections import defaultdict
from typing import Optional, List, Dict, Any

from Core.Common.Logger import logger
from arc_fuse_digimon.query_understanding import QueryUnderstanding, QueryAction
from arc_fuse_digimon.fusion import (
    rrf_fuse,
    normalize_graph_output,
    normalize_text_output,
)
from arc_fuse_digimon.disambiguation import EntityDisambiguation
from arc_fuse_digimon.evidence import EvidenceFusionModule
from arc_fuse_digimon.regeneration import ReGenerationAgent
from arc_fuse_digimon.commendor import Commendor
from arc_fuse_digimon.critic import CriticModule, CriticFeedback
from arc_fuse_digimon.normalizer import AnswerNormalizer


_route_stats = defaultdict(int)


def get_route_stats() -> dict:
    return dict(_route_stats)


def reset_route_stats():
    _route_stats.clear()


class ArcFuseEngine:
    """
    Unified engine for one fixed graph method + one fixed text method.

    QueryUnderstanding chooses among {graph, text, hybrid} for each query.
    Critic drives the iterative loop. Commendor is auxiliary and mostly used
    when the judge phase signals low-confidence evidence.
    """

    def __init__(
        self,
        graph_querier,
        text_retriever,
        entities_vdb=None,
        llm=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.graph_querier = graph_querier
        self.text_retriever = text_retriever
        self.entities_vdb = entities_vdb
        self.llm = llm or graph_querier.llm
        cfg = config or {}

        self.max_rounds = cfg.get("max_rounds", 3)
        self.use_routing = cfg.get("use_routing", True)
        self.use_regen = cfg.get("use_regen", True)
        self.use_critic = cfg.get("use_critic", True)
        self.use_commendor = cfg.get("use_commendor", True)
        self.use_normalizer = cfg.get("use_normalizer", True)
        self.use_disambiguation = cfg.get("use_disambiguation", True)
        self.top_k_fusion = cfg.get("top_k_fusion", 10)
        self.judge_score_threshold = float(cfg.get("judge_score_threshold", 3.0))
        self.critic_use_normalized_answer = cfg.get("critic_use_normalized_answer", True)
        self.use_last_resort_guess = cfg.get("use_last_resort_guess", False)

        self.query_understanding = QueryUnderstanding(self.llm)
        self.evidence_fusion = EvidenceFusionModule(self.llm)
        self.re_generator = ReGenerationAgent(
            self.llm,
            n_judges=cfg.get("n_judges", 3),
            n_voters=cfg.get("n_voters", 3),
        )
        self.critic = CriticModule(self.llm) if self.use_critic else None
        self.commendor = Commendor(self.llm) if self.use_commendor else None
        self.normalizer = AnswerNormalizer(self.llm) if self.use_normalizer else None
        self.disambiguator = (
            EntityDisambiguation(entities_vdb=self.entities_vdb)
            if self.use_disambiguation else None
        )

    async def query(self, question: str) -> Dict[str, Any]:
        """Run the full ARC-Fuse pipeline for a single question."""
        result = {
            "output": "",
            "selection_history": [],
            "rounds": 0,
            "commendor_decisions": [],
            "initial_selection": "",
            "domain": "",
            "initial_entities": [],
            "initial_relations": [],
            "route_source": "",
            "query_understanding_raw": "",
            "round_details": [],
        }

        if self.use_routing:
            action = await self.query_understanding.extract(question)
        else:
            action = QueryAction()
        action = self._ensure_route_source(
            action,
            default="llm" if self.use_routing else "routing_disabled",
        )

        result["initial_selection"] = action.selection
        result["domain"] = action.domain
        result["initial_entities"] = action.entities
        result["initial_relations"] = action.relations
        result["route_source"] = self._route_source(action)
        result["query_understanding_raw"] = action.raw

        answer = ""
        evidence_str = ""
        retrieval_query = question

        for round_idx in range(self.max_rounds):
            result["rounds"] = round_idx + 1
            result["selection_history"].append(action.selection)
            _route_stats[f"sel_{action.selection}"] += 1
            _route_stats[f"src_{self._route_source(action) or 'unknown'}"] += 1

            logger.info(
                f"[Round {round_idx + 1}/{self.max_rounds}] "
                f"selection={action.selection} query={retrieval_query[:120]}"
            )

            retriever_output = await self._retrieve(retrieval_query, action)
            retriever_output = await self._disambiguate(retriever_output)
            evidence_str, candidate = await self.evidence_fusion.fuse_and_generate(
                question, retriever_output
            )

            if self.use_regen:
                answer = await self.re_generator.generate(question, evidence_str, candidate)
                regen_debug = getattr(self.re_generator, "last_debug", {})
            else:
                answer = candidate
                regen_debug = {}

            answer_for_critic = await self._answer_for_feedback(question, answer)
            judge_scores = regen_debug.get("judge_scores", [])
            agg_score = float(regen_debug.get("agg_score", 0.0) or 0.0)
            low_confidence = bool(
                self.use_regen and judge_scores and agg_score < self.judge_score_threshold
            )

            critic_feedback = CriticFeedback(verdict="pass", confidence=1.0)
            critic_summary = "not_run"
            if self.critic:
                critic_feedback = await self.critic.evaluate(
                    question, evidence_str, answer_for_critic
                )
                critic_summary = (
                    f"verdict={critic_feedback.verdict}, "
                    f"{critic_feedback.action_summary()}"
                )
            bridge_entities = self._extract_bridge_entities(
                question=question,
                evidence=evidence_str,
                critic_feedback=critic_feedback,
                action=action,
            )

            round_detail = {
                "round": round_idx + 1,
                "selection": action.selection,
                "route_source": self._route_source(action),
                "retrieval_query": retrieval_query,
                "evidence_snippet": evidence_str[:400],
                "candidate": candidate,
                "intermediate_answer": answer,
                "normalized_for_critic": answer_for_critic,
                "judge_scores": judge_scores,
                "judge_candidates": regen_debug.get("judge_candidates", []),
                "agg_score": agg_score,
                "votes": regen_debug.get("votes", []),
                "low_confidence": low_confidence,
                "critic_verdict": critic_feedback.verdict,
                "critic_confidence": critic_feedback.confidence,
                "critic_actions": critic_feedback.action_summary(),
                "bridge_entities": bridge_entities,
                "commendor_kind": "",
                "round_outcome": "",
            }

            if self.critic and not critic_feedback.should_continue:
                pass_guard_reason = ""
                if round_idx < self.max_rounds - 1:
                    pass_guard_reason = self._pass_guard_followup(
                        question=question,
                        answer=answer_for_critic,
                        critic_feedback=critic_feedback,
                        bridge_entities=bridge_entities,
                    )

                if pass_guard_reason:
                    critic_feedback.verdict = "retrieve_more"
                    critic_feedback.confidence = min(critic_feedback.confidence, 0.5)
                    if not critic_feedback.suggestion:
                        critic_feedback.suggestion = pass_guard_reason
                    round_detail["critic_original_verdict"] = "pass"
                    round_detail["critic_verdict"] = critic_feedback.verdict
                    round_detail["critic_confidence"] = critic_feedback.confidence
                    round_detail["critic_actions"] = critic_feedback.action_summary()
                    round_detail["pass_guard_reason"] = pass_guard_reason
                    logger.info(f"Pass guard triggered: {pass_guard_reason}")
                else:
                    round_detail["commendor_kind"] = "pass(critic)"
                    round_detail["round_outcome"] = "stop_critic_pass"
                    result["round_details"].append(round_detail)
                    break

            if self.use_regen and self.critic and critic_feedback.verdict == "revise":
                retry_answer = await self.re_generator.re_generate_with_feedback(
                    question=question,
                    evidence=evidence_str,
                    prev_answer=answer,
                    feedback=critic_feedback.action_summary()
                    or "The previous answer did not fully match the evidence.",
                )
                retry_for_critic = await self._answer_for_feedback(question, retry_answer)
                retry_feedback = await self.critic.evaluate(
                    question, evidence_str, retry_for_critic
                )
                round_detail["regenerated_answer"] = retry_answer
                round_detail["regenerated_for_critic"] = retry_for_critic
                round_detail["regenerated_critic_verdict"] = retry_feedback.verdict
                round_detail["regenerated_critic_confidence"] = retry_feedback.confidence
                if not retry_feedback.should_continue:
                    answer = retry_answer
                    round_detail["intermediate_answer"] = retry_answer
                    round_detail["normalized_for_critic"] = retry_for_critic
                    round_detail["critic_verdict"] = retry_feedback.verdict
                    round_detail["critic_confidence"] = retry_feedback.confidence
                    round_detail["critic_actions"] = retry_feedback.action_summary()
                    round_detail["commendor_kind"] = "pass(revise)"
                    round_detail["round_outcome"] = "stop_revise_pass"
                    result["round_details"].append(round_detail)
                    break
                critic_feedback = retry_feedback
                critic_summary = (
                    f"verdict={critic_feedback.verdict}, "
                    f"{critic_feedback.action_summary()}"
                )

            decision = None
            if low_confidence and self.commendor:
                decision = await self.commendor.diagnose(
                    question=question,
                    selection=action.selection,
                    entities=action.entities_str(),
                    relations=action.relations_str(),
                    evidence=evidence_str,
                    answer=answer_for_critic,
                    critic_feedback=critic_summary,
                )
                result["commendor_decisions"].append({
                    "kind": decision.kind,
                    "confidence": decision.confidence,
                    "hint": decision.hint,
                })
                round_detail["commendor_kind"] = decision.kind
                round_detail["commendor_confidence"] = decision.confidence
                round_detail["commendor_hint"] = decision.hint

                if decision.is_pass:
                    round_detail["round_outcome"] = "stop_commendor_pass"
                    result["round_details"].append(round_detail)
                    break

                if decision.needs_regeneration and self.use_regen:
                    retry_answer = await self.re_generator.re_generate_with_feedback(
                        question=question,
                        evidence=evidence_str,
                        prev_answer=answer,
                        feedback=self._build_commendor_feedback(decision),
                    )
                    retry_for_critic = await self._answer_for_feedback(question, retry_answer)
                    retry_feedback = await self.critic.evaluate(
                        question, evidence_str, retry_for_critic
                    ) if self.critic else CriticFeedback()
                    round_detail["regenerated_answer"] = retry_answer
                    round_detail["regenerated_for_critic"] = retry_for_critic
                    round_detail["regenerated_critic_verdict"] = retry_feedback.verdict
                    round_detail["regenerated_critic_confidence"] = retry_feedback.confidence
                    if not retry_feedback.should_continue:
                        answer = retry_answer
                        round_detail["intermediate_answer"] = retry_answer
                        round_detail["normalized_for_critic"] = retry_for_critic
                        round_detail["critic_verdict"] = retry_feedback.verdict
                        round_detail["critic_confidence"] = retry_feedback.confidence
                        round_detail["critic_actions"] = retry_feedback.action_summary()
                        round_detail["round_outcome"] = "stop_commendor_regen_pass"
                        result["round_details"].append(round_detail)
                        break
                    critic_feedback = retry_feedback
                    critic_summary = (
                        f"verdict={critic_feedback.verdict}, "
                        f"{critic_feedback.action_summary()}"
                    )

            if round_idx >= self.max_rounds - 1:
                if not round_detail["commendor_kind"]:
                    round_detail["commendor_kind"] = "max_rounds"
                round_detail["round_outcome"] = "stop_max_rounds"
                result["round_details"].append(round_detail)
                break

            action = await self._next_action(
                question=question,
                action=action,
                critic_feedback=critic_feedback,
                decision=decision,
                selection_history=result["selection_history"],
                bridge_entities=bridge_entities,
            )
            retrieval_query = self._next_retrieval_query(
                question=question,
                critic_feedback=critic_feedback,
                decision=decision,
                bridge_entities=bridge_entities,
            )
            round_detail["next_selection"] = action.selection
            round_detail["next_retrieval_query"] = retrieval_query
            if not round_detail["commendor_kind"]:
                round_detail["commendor_kind"] = "critic_iterate"
            if decision and decision.needs_retriever_switch:
                round_detail["round_outcome"] = "continue_commendor_switch"
            elif decision and decision.needs_more_evidence:
                round_detail["round_outcome"] = "continue_commendor_more_evidence"
            elif decision and decision.needs_regeneration:
                round_detail["round_outcome"] = "continue_commendor_regen"
            elif round_detail.get("pass_guard_reason"):
                round_detail["round_outcome"] = "continue_pass_guard"
            else:
                round_detail["round_outcome"] = "continue_critic_iterate"
            result["round_details"].append(round_detail)

        bad_answers = {"unknown", "insufficient information", "insufficient information.", ""}
        if (
            self.use_last_resort_guess
            and self.use_regen
            and answer.strip().lower() in bad_answers
            and evidence_str
        ):
            try:
                answer = await self.re_generator.generate(
                    question,
                    evidence_str,
                    "",
                    feedback=(
                        "All retrieval rounds are exhausted. Make the best grounded guess "
                        "from the available evidence. Do not answer with insufficient information."
                    ),
                )
            except Exception as e:
                logger.warning(f"Last-resort inference failed: {e}")

        if self.normalizer and answer:
            try:
                normalized = await self.normalizer.normalize(question, answer)
                if normalized.strip().lower() != "unknown" or answer.strip().lower() == "unknown":
                    answer = normalized
            except Exception as e:
                logger.warning(f"Normalizer failed: {e}")

        result["output"] = answer
        return result

    async def _retrieve(self, question: str, action: QueryAction):
        """Call graph / text / hybrid retriever based on action.selection."""
        sel = action.selection

        if sel == "graph":
            return await self._graph_retrieve(question, action)

        if sel == "text":
            return await self._text_retrieve(question, action)

        graph_task = asyncio.create_task(self._graph_retrieve(question, action))
        text_task = asyncio.create_task(self._text_retrieve(question, action))
        graph_raw, text_raw = await asyncio.gather(
            graph_task, text_task, return_exceptions=True
        )
        if isinstance(graph_raw, Exception):
            logger.warning(f"Graph retrieval failed: {graph_raw}")
            graph_raw = None
        if isinstance(text_raw, Exception):
            logger.warning(f"Text retrieval failed: {text_raw}")
            text_raw = None

        graph_list = normalize_graph_output(graph_raw)
        text_list = normalize_text_output(text_raw)
        return rrf_fuse(graph_list, text_list, top_k=self.top_k_fusion)

    async def _graph_retrieve(self, question: str, action: QueryAction):
        """Use the configured graph querier to retrieve context."""
        try:
            if hasattr(self.graph_querier, "_retrieve_initialization"):
                await self.graph_querier._retrieve_initialization(question)
                max_depth = getattr(
                    getattr(self.graph_querier, "config", None), "depth", 3
                )
                for _ in range(max_depth):
                    await self.graph_querier._retrieve_relevant_contexts(
                        query=question,
                        mode="retrieve",
                    )
                    all_finish, _ = self.graph_querier._is_finish_list()
                    if all_finish:
                        break
                chains = self.graph_querier.reasoning_paths_list
                if not chains:
                    return None
                chain_text = "\n".join(
                    ", ".join(str(x) for x in chain)
                    for sublist in chains for chain in sublist
                )
                return (
                    f"Knowledge Triplets:\n{chain_text}"
                    if chain_text.strip() else None
                )
            return await self.graph_querier._retrieve_relevant_contexts(query=question)
        except Exception as e:
            logger.warning(f"Graph retrieve failed: {e}")
            return None

    async def _text_retrieve(self, question: str, action: QueryAction):
        """Use the configured text retriever (BM25 or VDB-based callable)."""
        if self.text_retriever is None:
            return None
        try:
            if hasattr(self.text_retriever, "retrieve"):
                return await self.text_retriever.retrieve(question)
            if callable(self.text_retriever):
                out = self.text_retriever(question)
                if asyncio.iscoroutine(out):
                    out = await out
                return out
        except Exception as e:
            logger.warning(f"Text retrieve failed: {e}")
            return None

    async def _disambiguate(self, retriever_output):
        """Apply entity disambiguation when structured entity-like output is present."""
        if not self.disambiguator:
            return retriever_output
        if not isinstance(retriever_output, list) or not retriever_output:
            return retriever_output

        mentions = []
        for doc in retriever_output:
            if isinstance(doc, dict) and doc.get("entity_name"):
                mentions.append(doc["entity_name"])
        if not mentions:
            return retriever_output

        try:
            resolved = await self.disambiguator.resolve(list(set(mentions)))
            return self.disambiguator.dedupe_by_canonical(retriever_output, resolved)
        except Exception as e:
            logger.warning(f"EntityDisambiguation failed: {e}")
            return retriever_output

    async def _answer_for_feedback(self, question: str, answer: str) -> str:
        """Normalize answers before Critic to reduce verbosity-related noise."""
        if not self.critic_use_normalized_answer or not self.normalizer or not answer:
            return answer
        try:
            return await self.normalizer.normalize(question, answer)
        except Exception as e:
            logger.warning(f"Feedback normalization failed: {e}")
            return answer

    async def _next_action(
        self,
        question: str,
        action: QueryAction,
        critic_feedback: CriticFeedback,
        decision,
        selection_history: List[str],
        bridge_entities: Optional[List[str]] = None,
    ) -> QueryAction:
        current_route_source = getattr(action, "route_source", "llm") or "llm"
        next_action = self._ensure_route_source(
            QueryAction(**action.to_dict()),
            default=current_route_source,
        )

        if decision and decision.needs_retriever_switch:
            feedback = self._build_commendor_feedback(decision)
            if self.use_routing:
                next_action = await self.query_understanding.extract(
                    question,
                    prev_action=action,
                    feedback=feedback,
                )
                next_action = self._ensure_route_source(next_action, default="llm")
            next_action = self._force_switch_if_same(next_action, selection_history)
            if bridge_entities and next_action.selection != "hybrid":
                next_action.selection = "hybrid"
                next_action.route_source = "bridge_override"
            return next_action

        if decision and decision.needs_more_evidence:
            if (
                next_action.selection != "hybrid"
                and self._same_selection_repeated(
                    selection_history,
                    next_action.selection,
                    min_repeats=2,
                )
            ):
                next_action.selection = "hybrid"
                next_action.route_source = "insufficient_evidence_escalation"
                logger.info(
                    "Escalated repeated insufficient evidence to hybrid retrieval"
                )
                return next_action

        if critic_feedback.broken_paths and next_action.selection == "text":
            next_action.selection = "hybrid"
            next_action.route_source = "critic_override"
        elif critic_feedback.conflicts and next_action.selection == "graph":
            next_action.selection = "hybrid"
            next_action.route_source = "critic_override"
        elif bridge_entities and critic_feedback.should_continue and next_action.selection != "hybrid":
            next_action.selection = "hybrid"
            next_action.route_source = "bridge_override"

        return next_action

    @staticmethod
    def _ensure_route_source(action: QueryAction, default: str = "llm") -> QueryAction:
        """Backfill route_source for older QueryAction definitions."""
        if not getattr(action, "route_source", ""):
            action.route_source = default
        return action

    @staticmethod
    def _route_source(action: QueryAction) -> str:
        return getattr(action, "route_source", "llm") or "llm"

    def _next_retrieval_query(
        self,
        question: str,
        critic_feedback: CriticFeedback,
        decision,
        bridge_entities: Optional[List[str]] = None,
    ) -> str:
        """Build the next retrieval query from structured Critic feedback."""
        parts: List[str] = []

        if critic_feedback.refined_query:
            parts.append(critic_feedback.refined_query.strip())
        else:
            parts.append(question.strip())
            if critic_feedback.missing_entities:
                parts.append(
                    "Focus on missing entities: " +
                    ", ".join(critic_feedback.missing_entities)
                )
            if critic_feedback.broken_paths:
                parts.append(
                    "Complete reasoning path: " +
                    "; ".join(critic_feedback.broken_paths)
                )
            if critic_feedback.conflicts:
                parts.append(
                    "Resolve ambiguity between: " +
                    "; ".join(critic_feedback.conflicts)
                )
            if critic_feedback.suggestion:
                parts.append(critic_feedback.suggestion.strip())

        if decision and decision.hint and not decision.needs_regeneration:
            parts.append(decision.hint.strip())

        bridge_entities = bridge_entities or []
        if bridge_entities:
            parts.append(
                "Second-hop anchors from previous evidence: "
                + ", ".join(bridge_entities[:4])
                + ". Retrieve facts connected to these anchors that answer the original question."
            )

        joined = " ".join(part for part in parts if part)
        return " ".join(joined.split()) or question

    def _extract_bridge_entities(
        self,
        question: str,
        evidence: str,
        critic_feedback: CriticFeedback,
        action: QueryAction,
        limit: int = 4,
    ) -> List[str]:
        """
        Extract second-hop anchors from the current evidence.

        A bridge entity is a salient proper noun/acronym that appears in the
        retrieved evidence but not in the original question. It is often the
        answer to the first hop in MuSiQue-style questions.
        """
        if not evidence:
            return []

        question_entities = set(self._candidate_entities(question))
        action_entities = set(getattr(action, "entities", []) or [])
        excluded = {self._entity_key(x) for x in question_entities | action_entities}

        critic_text = " ".join([
            " ".join(critic_feedback.missing_entities or []),
            " ".join(critic_feedback.broken_paths or []),
            " ".join(critic_feedback.conflicts or []),
            critic_feedback.suggestion or "",
            critic_feedback.refined_query or "",
        ])

        scores: Dict[str, float] = {}
        order: Dict[str, int] = {}
        for idx, ent in enumerate(self._candidate_entities(evidence[:3000])):
            key = self._entity_key(ent)
            if not key or key in excluded or self._is_bad_bridge_entity(ent):
                continue
            if key not in scores:
                scores[key] = 0.0
                order[key] = idx
            scores[key] += 1.0
            if ent.isupper() and len(ent) > 1:
                scores[key] += 1.5
            if ent.lower() in critic_text.lower():
                scores[key] += 1.0

        ranked = sorted(scores, key=lambda k: (-scores[k], order[k]))
        return [self._restore_entity_case(k, evidence) for k in ranked[:limit]]

    @staticmethod
    def _candidate_entities(text: str) -> List[str]:
        token = r"(?:[A-Z]{2,}|[A-Z][A-Za-z0-9&.'-]*)"
        pattern = re.compile(
            rf"\b{token}(?:\s+(?:(?:of|the|and|for|in|on|at|de|la|&)\s+)?{token})*\b"
        )
        entities = []
        for match in pattern.finditer(text or ""):
            ent = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:()[]{}\"'")
            if ent:
                entities.append(ent)

        # Graph retrievers often render triples as lower-case quoted strings:
        # ('ian ballantine', 'received graduate degree from', 'london school of economics').
        # The title-case pattern above misses those bridge entities, so recover
        # entity-like quoted subjects/objects while filtering obvious relations.
        for match in re.finditer(r"'([^']{2,100})'", text or ""):
            ent = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:()[]{}\"'")
            if ent and ArcFuseEngine._looks_like_quoted_entity(ent):
                entities.append(ent)
        return entities

    @staticmethod
    def _entity_key(entity: str) -> str:
        return re.sub(r"\s+", " ", entity.strip().lower())

    @staticmethod
    def _restore_entity_case(key: str, evidence: str) -> str:
        for ent in ArcFuseEngine._candidate_entities(evidence[:3000]):
            if ArcFuseEngine._entity_key(ent) == key:
                return ent
        return key

    @staticmethod
    def _is_bad_bridge_entity(entity: str) -> bool:
        ent = entity.strip()
        lower = ent.lower()
        bad_exact = {
            "the", "a", "an", "and", "or", "not", "none", "unknown",
            "insufficient information", "knowledge triplets", "candidate answer",
            "answer", "question", "retrieved context", "focus", "suggestion",
            "complete paths", "expand entities", "disambiguate",
            "american", "british", "english", "french", "german", "russian",
            "chinese", "japanese", "canadian", "australian", "national",
            "international",
        }
        if lower in bad_exact:
            return True
        if len(ent) <= 2 and not ent.isupper():
            return True
        if len(ent) > 70:
            return True
        if re.fullmatch(r"[A-Z][a-z]+", ent) and lower in {
            "the", "this", "that", "these", "those", "according", "it", "he",
            "she", "they", "in", "on", "at", "by", "for", "from", "with",
        }:
            return True
        if lower.startswith(("#", "round", "candidate", "answer")):
            return True
        if ArcFuseEngine._looks_like_relation_phrase(lower):
            return True
        return False

    @staticmethod
    def _looks_like_quoted_entity(text: str) -> bool:
        lower = text.strip().lower()
        if not lower:
            return False
        if len(lower.split()) > 7:
            return False
        if ArcFuseEngine._answer_category(lower) in {"temporal", "number", "ranking"}:
            return False
        if re.fullmatch(r"[\d\s.,:-]+", lower):
            return False
        if ArcFuseEngine._looks_like_relation_phrase(lower):
            return False
        return True

    @staticmethod
    def _looks_like_relation_phrase(text: str) -> bool:
        lower = text.strip().lower()
        relation_markers = (
            " is ", " was ", " are ", " were ", " has ", " had ", " have ",
            "fought between", "received ", "operated ", "located ", "defeated ",
            "ruled", "born", "died", "borders", "population", "has area",
            "part of", "sings", "hosted by", "written by", "directed by",
            "released", "aired", "broadcasts", "plays", "member of",
        )
        relation_prefixes = (
            "is ", "was ", "are ", "were ", "has ", "had ", "have ",
            "received ", "operated ", "located ", "defeated ", "ruled ",
            "borders ", "population ", "released ", "aired ", "broadcasts ",
        )
        return (
            any(marker in lower for marker in relation_markers)
            or any(lower.startswith(prefix) for prefix in relation_prefixes)
        )

    def _pass_guard_followup(
        self,
        question: str,
        answer: str,
        critic_feedback: CriticFeedback,
        bridge_entities: Optional[List[str]],
    ) -> str:
        """
        Prevent premature Critic pass on compositional questions.

        MuSiQue-style questions often retrieve a first-hop bridge entity and
        then need one more lookup. If the Critic says pass while the answer
        looks like that bridge, or the Critic still reports retrieval gaps, run
        another targeted round instead of stopping early.
        """
        if not self._looks_multihop_question(question):
            return ""

        normalized_answer = self._normalize_match_text(answer)
        if normalized_answer in {
            "unknown", "insufficient information", "insufficient information."
        }:
            return "Pass guard: answer is still insufficient on a multi-hop question."

        bridge_entities = bridge_entities or []
        answer_matches_bridge = self._answer_matches_any_entity(answer, bridge_entities)
        if answer_matches_bridge and self._answer_needs_bridge_followup(question):
            return (
                "Pass guard: answer matches a likely first-hop bridge entity on "
                "a bridge-style multi-hop question; retrieve the next-hop fact."
            )

        if (
            answer_matches_bridge
            and not self._answer_looks_like_final_for_question(question, answer)
        ):
            return (
                "Pass guard: answer matches a likely first-hop bridge entity; "
                "retrieve the next-hop fact."
            )

        if (
            critic_feedback.confidence <= 0.8
            and self._critic_reports_retrieval_gap(critic_feedback)
        ):
            return (
                "Pass guard: Critic marked pass but still reported missing "
                "entities, broken paths, or retrieval suggestions."
            )

        return ""

    @staticmethod
    def _looks_multihop_question(question: str) -> bool:
        q_lower = question.lower()
        markers = (
            " where the ", " where ", " in the state where",
            " in the country where", " in the continent where",
            " country where", " state where", " continent where",
            " city where", " county where", " district of the country",
            " related to", " associated with", " named after",
            " beaten at", " aired ", " that aired", " who sang",
            " who wrote", " who directed", " whose ", " of the country",
            " of the state", " the state whose", " the country whose",
        )
        return any(marker in q_lower for marker in markers)

    @staticmethod
    def _answer_needs_bridge_followup(question: str) -> bool:
        """
        Detect bridge-style multi-hop questions where a named-entity answer can
        still be only the first hop.

        For example, "who was in charge of the country beaten at X" can return
        a historical battle participant. It has the right surface type for
        "who", but it is still not the requested leader of the resolved country.
        """
        q_lower = question.lower()
        bridge_markers = (
            " of the country ", " of the state ", " of the city ",
            " of the county ", " of the university ", " of the school ",
            " country where", " state where", " city where", " county where",
            " related to", " associated with", " named after", " beaten at",
            " defeated at", " beaten by", " defeated by",
        )
        return any(marker in f" {q_lower} " for marker in bridge_markers)

    @classmethod
    def _answer_looks_like_final_for_question(cls, question: str, answer: str) -> bool:
        """Heuristic answer-type check for bridge-entity pass guarding."""
        q_lower = question.lower()
        a_lower = str(answer or "").strip().lower()
        category = cls._answer_category(answer)

        if cls._asks_when(q_lower):
            return category == "temporal"
        if cls._asks_numeric(q_lower):
            return category == "number"
        if cls._asks_ranking(q_lower):
            return category == "ranking"

        if " instance of" in q_lower or "what type" in q_lower or "what kind" in q_lower:
            return not cls._looks_like_named_entity_answer(answer)

        if "who" in q_lower:
            return category not in {"temporal", "number", "ranking"}

        if any(marker in q_lower for marker in (
            "what county", "which county", "administrative territory",
            "what district", "which district", "what region", "which region",
        )):
            return any(marker in a_lower for marker in (
                "county", "district", "region", "province", "state",
                "territory", "municipality", "parish", "borough", "prefecture",
                "central", "northern", "southern", "eastern", "western",
                "north", "south", "east", "west",
            ))

        if "where" in q_lower or "located" in q_lower:
            organization_markers = (
                "school", "university", "college", "institute", "ministry",
                "records", "record", "network", "corps", "party", "club",
                "ravens", "49ers", "falcons", "panthers",
            )
            return not any(marker in a_lower for marker in organization_markers)

        return True

    @staticmethod
    def _asks_when(q_lower: str) -> bool:
        return any(marker in q_lower for marker in (
            "when", "what date", "which date", "what year",
        ))

    @staticmethod
    def _asks_numeric(q_lower: str) -> bool:
        return any(marker in q_lower for marker in (
            "how many", "population", "how old", "age", "number of",
        ))

    @staticmethod
    def _asks_ranking(q_lower: str) -> bool:
        return any(marker in q_lower for marker in (
            "ranking", "rank", "largest", "smallest", "highest", "lowest",
            "biggest", "size ranking",
        ))

    @classmethod
    def _answer_category(cls, answer: str) -> str:
        text = str(answer or "").strip()
        lower = text.lower()
        if not text:
            return "empty"
        if lower in {"unknown", "insufficient information", "insufficient information."}:
            return "insufficient"
        if cls._looks_temporal_text(text):
            return "temporal"
        if re.search(r"\b\d[\d,]*(?:\.\d+)?\b", text):
            return "number"
        if re.search(
            r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
            r"ninth|tenth|\d+(?:st|nd|rd|th)?)[ -]?"
            r"(?:largest|smallest|highest|lowest|biggest|oldest|youngest)\b",
            lower,
        ) or lower in {"largest", "smallest", "highest", "lowest", "biggest"}:
            return "ranking"
        return "entity_or_phrase"

    @staticmethod
    def _looks_temporal_text(text: str) -> bool:
        month = (
            r"january|february|march|april|may|june|july|august|september|"
            r"october|november|december|jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|"
            r"jul\.?|aug\.?|sep\.?|sept\.?|oct\.?|nov\.?|dec\.?"
        )
        lower = text.lower()
        return bool(
            re.search(rf"\b(?:{month})\s+\d{{1,2}}(?:,|\s+of)?\s+\d{{4}}\b", lower)
            or re.search(r"\b\d{4}\b", lower)
            or re.search(r"\b(?:early|mid|late)[ -]?\d{1,2}(?:st|nd|rd|th)\s+centur", lower)
            or re.search(r"\b\d{1,2}(?:st|nd|rd|th)\s+centur", lower)
        )

    @staticmethod
    def _looks_like_named_entity_answer(answer: str) -> bool:
        text = str(answer or "").strip()
        if not text:
            return False
        return bool(
            re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
            or re.search(r"\b[A-Z]{2,}\b", text)
        )

    @staticmethod
    def _critic_reports_retrieval_gap(critic_feedback: CriticFeedback) -> bool:
        if (
            critic_feedback.missing_entities
            or critic_feedback.broken_paths
            or critic_feedback.conflicts
        ):
            return True
        text = " ".join([
            critic_feedback.suggestion or "",
            critic_feedback.refined_query or "",
        ]).lower()
        negative_markers = (
            "no additional", "no further", "not needed", "nothing additional",
            "current evidence is sufficient", "answer is sufficient",
            "well-supported",
        )
        if any(marker in text for marker in negative_markers):
            return False
        positive_markers = (
            "specific", "missing", "complete", "ranking", "rankings",
            "leadership", "leader", "minister", "mayor", "president",
            "timeline", "currency", "pledge", "population", "monsoon",
            "birthplace", "specific location", "broadcasts", "relevance",
            "national anthem", "administrative territory",
        )
        return any(marker in text for marker in positive_markers)

    @classmethod
    def _answer_matches_any_entity(cls, answer: str, entities: List[str]) -> bool:
        answer_key = cls._normalize_match_text(answer)
        if not answer_key:
            return False
        answer_tokens = cls._match_tokens(answer_key)
        for entity in entities or []:
            entity_key = cls._normalize_match_text(entity)
            if not entity_key:
                continue
            if answer_key in entity_key or entity_key in answer_key:
                return True
            entity_tokens = cls._match_tokens(entity_key)
            if answer_tokens and entity_tokens:
                overlap = len(answer_tokens & entity_tokens) / max(1, len(answer_tokens))
                if overlap >= 0.7:
                    return True
        return False

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(text or ""))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-zA-Z0-9]+", " ", normalized.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _match_tokens(text: str) -> set:
        return {token for token in text.split() if len(token) > 2}

    @staticmethod
    def _build_commendor_feedback(decision) -> str:
        if not decision:
            return ""
        parts = []
        if decision.reason:
            parts.append(decision.reason.strip())
        if decision.hint:
            parts.append(f"HINT: {decision.hint.strip()}")
        return " ".join(parts).strip()

    @staticmethod
    def _force_switch_if_same(action: QueryAction, history: List[str]) -> QueryAction:
        """If reflection returns the same selection, force a different retriever."""
        if not history:
            return action
        last = history[-1]
        if action.selection != last:
            return action
        order = {"graph": "text", "text": "hybrid", "hybrid": "graph"}
        action.selection = order.get(last, "hybrid")
        logger.info(f"Forced selection switch: {last} -> {action.selection}")
        return action

    @staticmethod
    def _same_selection_repeated(
        history: List[str],
        selection: str,
        min_repeats: int = 2,
    ) -> bool:
        """Return True when the same retriever has just failed repeatedly."""
        if not history or len(history) < min_repeats:
            return False
        tail = history[-min_repeats:]
        return all(item == selection for item in tail)
