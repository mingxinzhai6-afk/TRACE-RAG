"""
QueryUnderstanding for ARC-Fuse.

Following Agent-G (ICLR 2025) Section 2.2:
    a_t = {d_t, E_t(d_t), R_t(d_t), s_t}
where
    d_t : domain (e.g., sports, movie, finance, science, biography, general)
    E_t : topic entities (with categories)
    R_t : useful relations
    s_t : retriever selection ∈ {graph, text, hybrid}

If feedback from Commendor is provided (iteration > 1), uses the reflection
prompt to refine the action.

Usage:
    qu = QueryUnderstanding(llm)
    action = await qu.extract(question)
    # action = QueryAction(domain=..., entities=[...], relations=[...], selection=...)

    # With reflection:
    action = await qu.extract(question, prev_action=action, feedback="...")
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from Core.Common.Logger import logger
from Core.Common.Utils import prase_json_from_response
from arc_fuse_digimon.prompts import (
    QUERY_UNDERSTANDING_PROMPT,
    QUERY_UNDERSTANDING_REFLECT_PROMPT,
)


@dataclass
class QueryAction:
    """Structured action a_t."""
    domain: str = "general"
    entities: List[str] = field(default_factory=list)           # clean names (no category)
    entities_annotated: List[str] = field(default_factory=list) # with "(category)" kept
    relations: List[str] = field(default_factory=list)
    selection: str = "graph"  # graph | text | hybrid
    raw: str = ""             # raw LLM output for debugging
    route_source: str = "llm"

    def to_dict(self) -> dict:
        return asdict(self)

    def entities_str(self) -> str:
        return ", ".join(self.entities_annotated) if self.entities_annotated else "none"

    def relations_str(self) -> str:
        return ", ".join(self.relations) if self.relations else "none"


class QueryUnderstanding:
    """LLM-powered query understanding module."""

    VALID_SELECTIONS = {"graph", "text", "hybrid"}

    def __init__(self, llm):
        self.llm = llm

    async def extract(
        self,
        question: str,
        prev_action: Optional[QueryAction] = None,
        feedback: str = "",
    ) -> QueryAction:
        """
        Extract the action a_t for a question.

        If prev_action + feedback are provided, runs the reflection prompt to
        refine the action.
        """
        if prev_action is not None and feedback:
            prompt = QUERY_UNDERSTANDING_REFLECT_PROMPT.format(
                question=question,
                prev_domain=prev_action.domain,
                prev_entities=prev_action.entities_str(),
                prev_relations=prev_action.relations_str(),
                prev_selection=prev_action.selection,
                feedback=feedback,
            )
        else:
            prompt = QUERY_UNDERSTANDING_PROMPT.format(question=question)

        try:
            response = await self.llm.aask(msg=prompt)
        except Exception as e:
            logger.warning(f"QueryUnderstanding LLM call failed: {e}")
            return QueryAction(domain="general", selection="hybrid")

        action = self._parse(response, question)
        logger.info(
            f"QueryUnderstanding: domain={action.domain}, "
            f"entities={action.entities}, relations={action.relations}, "
            f"selection={action.selection}"
        )
        return action

    @classmethod
    def _parse(cls, response: str, question: str = "") -> QueryAction:
        """Parse LLM free-form output into a QueryAction."""
        domain = "general"
        entities_annotated: List[str] = []
        relations: List[str] = []
        selection = ""
        route_source = "llm"

        parsed_json = None
        try:
            parsed_json = prase_json_from_response(response)
        except Exception:
            parsed_json = None

        if isinstance(parsed_json, dict):
            raw_domain = str(parsed_json.get("domain", "general")).strip().lower()
            if raw_domain and raw_domain != "none":
                domain = raw_domain.split()[0]

            raw_entities = parsed_json.get("entities", parsed_json.get("topic_entities", []))
            if isinstance(raw_entities, str):
                raw_entities = [e.strip() for e in raw_entities.split(",") if e.strip()]
            if isinstance(raw_entities, list):
                entities_annotated = [str(e).strip() for e in raw_entities if str(e).strip()]

            raw_relations = parsed_json.get("relations", parsed_json.get("useful_relations", []))
            if isinstance(raw_relations, str):
                raw_relations = [r.strip() for r in raw_relations.split(",") if r.strip()]
            if isinstance(raw_relations, list):
                relations = [str(r).strip() for r in raw_relations if str(r).strip()]

            raw_selection = str(parsed_json.get("selection", "")).strip().lower().strip('"').strip("'")
            if raw_selection in cls.VALID_SELECTIONS:
                selection = raw_selection

        if not selection:
            for line in response.strip().split("\n"):
                low = line.strip().lower()
                if low.startswith("domain:"):
                    v = line.split(":", 1)[1].strip().lower()
                    if v and v != "none":
                        domain = v.split()[0]
                elif low.startswith("topic entities:"):
                    raw = line.split(":", 1)[1].strip()
                    if raw and raw.lower() != "none":
                        entities_annotated = [e.strip() for e in raw.split(",") if e.strip()]
                elif low.startswith("useful relations:"):
                    raw = line.split(":", 1)[1].strip()
                    if raw and raw.lower() != "none":
                        relations = [r.strip() for r in raw.split(",") if r.strip()]
                elif low.startswith("selection:"):
                    v = line.split(":", 1)[1].strip().lower().strip('"').strip("'")
                    if "hybrid" in v:
                        selection = "hybrid"
                    elif "text" in v or "document" in v:
                        selection = "text"
                    elif "graph" in v:
                        selection = "graph"

        # Strip "(category)" for clean entity names
        entities_clean = []
        for e in entities_annotated:
            clean = re.sub(r"\s*\([^)]*\)\s*$", "", e).strip()
            if clean:
                entities_clean.append(clean)
        heuristic_selection = cls._heuristic_selection(
            question=question,
            entities=entities_clean,
            relations=relations,
        )
        if selection not in cls.VALID_SELECTIONS:
            selection = heuristic_selection
            route_source = "heuristic"
        elif heuristic_selection != selection and cls._should_override_llm(
            question=question,
            entities=entities_clean,
            llm_selection=selection,
            heuristic_selection=heuristic_selection,
        ):
            selection = heuristic_selection
            route_source = "heuristic_override"

        return QueryAction(
            domain=domain,
            entities=entities_clean,
            entities_annotated=entities_annotated,
            relations=relations,
            selection=selection,
            raw=response,
            route_source=route_source,
        )

    @classmethod
    def _should_override_llm(
        cls,
        question: str,
        entities: List[str],
        llm_selection: str,
        heuristic_selection: str,
    ) -> bool:
        """Apply a light-weight override when the LLM selection is clearly implausible."""
        if llm_selection == heuristic_selection:
            return False
        if not entities and llm_selection == "graph":
            return True
        q_lower = question.lower()
        if cls._looks_descriptive(q_lower) and llm_selection == "graph":
            return True
        if cls._looks_compositional(q_lower) and llm_selection == "graph":
            return True
        return False

    @classmethod
    def _heuristic_selection(
        cls,
        question: str,
        entities: List[str],
        relations: List[str],
    ) -> str:
        """Fallback routing aligned with the paper's graph/text/hybrid split."""
        q_lower = question.lower().strip()

        if cls._looks_comparative(q_lower) or cls._looks_compositional(q_lower):
            return "hybrid"

        if not entities:
            return "text"

        if cls._looks_descriptive(q_lower):
            return "text"

        if relations:
            return "graph"

        attribute_markers = (
            "'s occupation", "'s birthplace", "'s birth place", "'s birthdate",
            "'s father", "'s mother", "'s spouse", "'s network", "'s county",
            "'s capital", "'s population", "who directed", "who wrote",
            "who plays", "when did", "where was", "where is", "what is the occupation",
            "what network", "what county", "what city", "what country",
        )
        if any(marker in q_lower for marker in attribute_markers):
            return "graph"

        return "graph"

    @staticmethod
    def _looks_descriptive(q_lower: str) -> bool:
        descriptive_markers = (
            "how does", "how do", "why does", "why do", "explain",
            "describe", "background of", "history of", "what is a ",
            "what is an ", "what are ", "tell me about",
        )
        if any(marker in q_lower for marker in descriptive_markers):
            return True
        if q_lower.startswith("what is ") and "'s " not in q_lower and " where " not in q_lower:
            return True
        return False

    @staticmethod
    def _looks_comparative(q_lower: str) -> bool:
        comparative_markers = (
            "older", "younger", "larger", "smaller", "taller", "shorter",
            "compare", "comparison", "difference", "versus", " vs ",
            "how many", "count the", "list all", "list the",
        )
        return any(marker in q_lower for marker in comparative_markers)

    @staticmethod
    def _looks_compositional(q_lower: str) -> bool:
        compositional_markers = (
            "same state as", "same country as", "same county as", "same artist as",
            "same network as", "named after", "part named after",
            "in the state where", "in the country where", "in the county where",
            "of the country where", "of the state where", "of the place where",
            "where the ", "who plays the", "the birthplace of", "the place where",
        )
        return any(marker in q_lower for marker in compositional_markers)
