"""
EntityDisambiguation — alias/variant resolution for entities returned by retrievers.

Takes a list of entity-like items (from graph retrieval) or a list of
mentioned entities (extracted by QueryUnderstanding), and groups them by
likely real-world identity using VDB embedding similarity (preferred) with
a string-similarity fallback.

Design choice: VDB embedding (option C) — consistent with the existing
DIGIMON codebase where entity linking already uses VDB.

Usage:
    disambiguator = EntityDisambiguation(entities_vdb=vdb, threshold=0.82)
    resolved = await disambiguator.resolve(["Barack Obama", "Obama"])
    # Returns: {"Barack Obama": "Barack Obama", "Obama": "Barack Obama"}
"""

import difflib
from typing import List, Dict, Optional
from Core.Common.Logger import logger


class EntityDisambiguation:
    """
    Resolve entity aliases/variants using VDB embedding similarity.

    If entities_vdb is provided, uses the VDB's nearest-neighbor lookup to
    map each input name to its canonical graph entity. Otherwise, falls back
    to case-insensitive fuzzy string matching.
    """

    def __init__(
        self,
        entities_vdb=None,
        threshold: float = 0.82,
        string_sim_threshold: float = 0.85,
    ):
        """
        Args:
            entities_vdb: DIGIMON EntityVectorDB (has retrieval_nodes). If None,
                          falls back to string similarity.
            threshold: VDB score threshold for accepting a match.
            string_sim_threshold: SequenceMatcher ratio threshold for the
                                   string fallback.
        """
        self.entities_vdb = entities_vdb
        self.threshold = threshold
        self.string_sim_threshold = string_sim_threshold

    async def resolve(self, mentions: List[str]) -> Dict[str, str]:
        """
        Map each mention to its canonical form.

        Returns:
            dict {mention -> canonical_name}. Mentions with no match map to
            themselves.
        """
        if not mentions:
            return {}

        resolved: Dict[str, str] = {}

        if self.entities_vdb is not None:
            for m in mentions:
                try:
                    canonical = await self._vdb_lookup(m)
                    resolved[m] = canonical or m
                except Exception as e:
                    logger.warning(f"EntityDisambiguation VDB lookup failed for '{m}': {e}")
                    resolved[m] = m
        else:
            # String-similarity cluster among the mentions themselves
            clusters = self._cluster_by_string_sim(mentions)
            for cluster in clusters:
                canonical = sorted(cluster, key=lambda s: -len(s))[0]
                for m in cluster:
                    resolved[m] = canonical

        return resolved

    async def resolve_and_group(self, mentions: List[str]) -> List[List[str]]:
        """
        Return mentions grouped into alias clusters.

        Example input: ["Obama", "Barack Obama", "Merkel"]
        Example output: [["Obama", "Barack Obama"], ["Merkel"]]
        """
        resolved = await self.resolve(mentions)
        groups: Dict[str, List[str]] = {}
        for m in mentions:
            c = resolved.get(m, m)
            groups.setdefault(c, []).append(m)
        return list(groups.values())

    async def _vdb_lookup(self, mention: str) -> Optional[str]:
        """Look up a mention in the entity VDB; return canonical name on match."""
        try:
            nodes = await self.entities_vdb.retrieval_nodes(
                query=mention, top_k=1, graph=None
            )
        except TypeError:
            nodes = await self.entities_vdb.retrieval_nodes(mention, 1, None)
        except Exception as e:
            logger.warning(f"VDB retrieval_nodes failed: {e}")
            return None

        if not nodes:
            return None

        first = nodes[0]
        if isinstance(first, dict):
            name = first.get("entity_name") or first.get("name")
            score = first.get("score") or first.get("similarity") or 1.0
        else:
            name = getattr(first, "entity_name", None) or getattr(first, "name", None)
            score = getattr(first, "score", 1.0)

        if name and score >= self.threshold:
            return name
        return None

    def _cluster_by_string_sim(self, mentions: List[str]) -> List[List[str]]:
        """Greedy clustering by SequenceMatcher ratio."""
        clusters: List[List[str]] = []
        for m in mentions:
            placed = False
            for cluster in clusters:
                if any(
                    difflib.SequenceMatcher(None, m.lower(), c.lower()).ratio()
                    >= self.string_sim_threshold
                    for c in cluster
                ):
                    cluster.append(m)
                    placed = True
                    break
            if not placed:
                clusters.append([m])
        return clusters

    @staticmethod
    def dedupe_by_canonical(docs: List[Dict], resolved: Dict[str, str],
                             name_field: str = "entity_name") -> List[Dict]:
        """
        Deduplicate a list of entity-like docs by their canonical name.

        Keeps the highest-ranked doc per canonical group.
        """
        seen = set()
        out = []
        for doc in docs:
            name = doc.get(name_field, "")
            canonical = resolved.get(name, name)
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append(doc)
        return out
