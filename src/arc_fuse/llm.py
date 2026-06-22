from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OpenAICompatibleLLM:
    api_key: str
    base_url: str
    model: str
    timeout: float = 120.0
    temperature: float = 0.0

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleLLM":
        api_key = os.environ.get("ARC_FUSE_API_KEY", "").strip()
        base_url = os.environ.get("ARC_FUSE_BASE_URL", "").strip()
        model = os.environ.get("ARC_FUSE_MODEL", "").strip()
        missing = [
            name
            for name, value in (
                ("ARC_FUSE_API_KEY", api_key),
                ("ARC_FUSE_BASE_URL", base_url),
                ("ARC_FUSE_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        return cls(api_key=api_key, base_url=base_url, model=model)

    async def complete(self, prompt: str) -> str:
        return await asyncio.to_thread(self._complete_sync, prompt)

    def _complete_sync(self, prompt: str) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM endpoint request failed: {exc.reason}") from exc

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected chat-completions response shape") from exc


class ScriptedLLM:
    """Deterministic offline test double for the bundled synthetic example."""

    async def complete(self, prompt: str) -> str:
        if "<task:route>" in prompt:
            return self._route(prompt)
        if "<task:judge>" in prompt:
            answer = self._answer(prompt)
            score = 9 if answer != "unknown" else 2
            return json.dumps(
                {
                    "score": score,
                    "reason": "synthetic evidence match",
                    "candidate": answer,
                }
            )
        if "<task:vote>" in prompt:
            answer = self._answer(prompt)
            return json.dumps(
                {"vote": answer, "reason": "synthetic majority vote"}
            )
        if "<task:critic>" in prompt:
            answer = self._field(prompt, "Current Answer:", "Schema:")
            evidence = self._field(prompt, "Evidence:", "Current Answer:")
            passed = (
                bool(answer)
                and answer.lower() != "unknown"
                and answer.lower() in evidence.lower()
            )
            if passed:
                return json.dumps(
                    {
                        "verdict": "pass",
                        "confidence": 0.95,
                        "feedback": {
                            "missing_entities": [],
                            "broken_paths": [],
                            "conflicts": [],
                            "suggestion": "",
                        },
                        "refined_query": "",
                    }
                )
            question = self._field(prompt, "Question:", "Evidence:")
            return json.dumps(
                {
                    "verdict": "retrieve_more",
                    "confidence": 0.8,
                    "feedback": {
                        "missing_entities": [],
                        "broken_paths": [],
                        "conflicts": [],
                        "suggestion": "retrieve evidence directly matching the question",
                    },
                    "refined_query": question,
                }
            )
        if "<task:commendor>" in prompt:
            answer = self._field(prompt, "Current Answer:", "Critic feedback:")
            decision = "insufficient_evidence" if answer == "unknown" else "pass"
            return json.dumps(
                {
                    "decision": decision,
                    "confidence": 0.9,
                    "reason": "offline diagnostic",
                    "hint": "broaden retrieval" if decision != "pass" else "",
                }
            )
        if "<task:normalize>" in prompt:
            answer = self._field(prompt, "Raw Answer:", "Normalized Answer:")
            return self._clean_answer(answer)
        if "<task:candidate>" in prompt or "<task:infer>" in prompt:
            return self._answer(prompt)
        return "unknown"

    def _route(self, prompt: str) -> str:
        question = self._field(prompt, "Question:", "Schema:")
        lower = question.lower()
        selection = "graph"
        if any(marker in lower for marker in ("explain", "describe", "why", "how does")):
            selection = "text"
        elif any(marker in lower for marker in ("compare", "same", "where the")):
            selection = "hybrid"
        entities = re.findall(
            r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b",
            question,
        )
        entities = [
            entity
            for entity in entities
            if entity.lower() not in {"what", "who", "where", "when", "which", "in"}
        ]
        return json.dumps(
            {
                "domain": "general",
                "entities": entities,
                "relations": [],
                "selection": selection,
            }
        )

    def _answer(self, prompt: str) -> str:
        question = self._field(prompt, "Question:", "Evidence:")
        evidence = self._field(
            prompt,
            "Evidence:",
            "Judge outputs:",
            "Aggregated score:",
            "Previous answer:",
            "Candidate Answer:",
        )
        q = question.lower()

        if "capital of france" in q:
            match = re.search(r"([A-Za-z ]+?)\s+--capital_of-->\s+France", evidence)
            if match:
                return self._clean_answer(match.group(1))
            match = re.search(r"capital(?: and largest city)? is ([A-Z][A-Za-z ]+)", evidence)
            if match:
                return self._clean_answer(match.group(1))

        if "wrote pride and prejudice" in q:
            match = re.search(
                r"([A-Za-z ]+?)\s+--wrote-->\s+Pride and Prejudice",
                evidence,
            )
            if match:
                return self._clean_answer(match.group(1))
            match = re.search(r"written by ([A-Z][A-Za-z ]+)", evidence)
            if match:
                return self._clean_answer(match.group(1))

        if "eiffel tower" in q and ("city" in q or "located" in q):
            match = re.search(
                r"Eiffel Tower\s+--located_in-->\s+([A-Za-z ]+)",
                evidence,
            )
            if match:
                return self._clean_answer(match.group(1))
            match = re.search(r"located in ([A-Z][A-Za-z]+)", evidence)
            if match:
                return self._clean_answer(match.group(1))

        winning = self._field(prompt, "Winning answer:", "Final Answer:")
        return self._clean_answer(winning) if winning else "unknown"

    @staticmethod
    def _field(prompt: str, start: str, *end_markers: str) -> str:
        if start not in prompt:
            return ""
        value = prompt.split(start, 1)[1]
        positions = [
            value.find(marker)
            for marker in end_markers
            if marker and value.find(marker) >= 0
        ]
        if positions:
            value = value[: min(positions)]
        return value.strip()

    @staticmethod
    def _clean_answer(value: str) -> str:
        text = (value or "").strip().splitlines()[0] if value else ""
        text = text.strip(" \t\r\n\"'.:-")
        text = re.sub(r"^\[[^\]]+\]\s*", "", text)
        return text or "unknown"
