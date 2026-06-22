from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .models import QueryAction


class AsyncLLM(Protocol):
    async def complete(self, prompt: str) -> str:
        """Return one model completion for the supplied prompt."""


class AsyncRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        action: QueryAction,
        top_k: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Return ranked evidence dictionaries."""
