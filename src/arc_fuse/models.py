from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PipelineConfig:
    max_rounds: int = 3
    top_k: int = 5
    rrf_k: int = 60
    graph_weight: float = 1.0
    text_weight: float = 1.0
    n_judges: int = 3
    n_voters: int = 3
    judge_score_threshold: float = 3.0
    use_routing: bool = True
    use_regeneration: bool = True
    use_critic: bool = True
    use_commendor: bool = True
    use_normalizer: bool = True
    use_disambiguation: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PipelineConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass
class QueryAction:
    domain: str = "general"
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    selection: str = "hybrid"
    route_source: str = "llm"
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriticFeedback:
    verdict: str = "pass"
    confidence: float = 1.0
    missing_entities: list[str] = field(default_factory=list)
    broken_paths: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    suggestion: str = ""
    refined_query: str = ""

    @property
    def should_continue(self) -> bool:
        return self.verdict in {"retrieve_more", "revise"}

    def action_summary(self) -> str:
        parts: list[str] = []
        if self.missing_entities:
            parts.append("expand " + ", ".join(self.missing_entities))
        if self.broken_paths:
            parts.append("complete " + ", ".join(self.broken_paths))
        if self.conflicts:
            parts.append("resolve " + ", ".join(self.conflicts))
        if self.suggestion:
            parts.append(self.suggestion)
        return "; ".join(parts) or "no action"


@dataclass
class CommendorDecision:
    kind: str = "pass"
    confidence: float = 1.0
    reason: str = ""
    hint: str = ""


@dataclass
class PipelineResult:
    question: str
    answer: str
    initial_action: QueryAction
    selection_history: list[str]
    rounds: int
    round_details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["initial_action"] = self.initial_action.to_dict()
        return data
