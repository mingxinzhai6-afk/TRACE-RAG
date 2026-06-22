from __future__ import annotations

import difflib
import re
from typing import Any, Mapping, Sequence


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def deduplicate_entities(
    results: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.88,
) -> list[dict[str, Any]]:
    """Keep the highest-ranked result from each likely entity alias group."""
    kept: list[dict[str, Any]] = []
    canonical_names: list[str] = []

    for raw_result in results:
        result = dict(raw_result)
        name = str(result.get("entity_name", "")).strip()
        if not name:
            kept.append(result)
            continue
        key = _canonical_key(name)
        duplicate = any(
            key == existing
            or difflib.SequenceMatcher(None, key, existing).ratio() >= threshold
            for existing in canonical_names
        )
        if duplicate:
            continue
        canonical_names.append(key)
        kept.append(result)
    return kept
