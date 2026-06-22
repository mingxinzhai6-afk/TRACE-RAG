"""
Answer Normalizer Module (Innovation 2)

After generation_qa() returns a possibly verbose answer, this module
normalizes it into a concise benchmark-aligned form.

The normalizer is intentionally conservative for benchmark-fragile answers:
exact dates, ranges, ordinal rankings, and qualified place/organization names
are preserved when the LLM normalization would make them less specific.
"""

import re

from Core.Common.Logger import logger
from arc_fuse_digimon.prompts import ANSWER_NORMALIZER_PROMPT


class AnswerNormalizer:
    def __init__(self, llm):
        self.llm = llm

    async def normalize(self, question: str, raw_answer: str) -> str:
        """
        Normalize a verbose QA answer into a concise benchmark-aligned form.
        """
        if not raw_answer or raw_answer.strip() == "":
            return "unknown"

        # Step 1: rule-based pre-extraction for CoT-style and fragile answers.
        pre_extracted = self._rule_extract(question, raw_answer)

        # Step 2: LLM-based normalization, then restore fragile specificity.
        prompt = ANSWER_NORMALIZER_PROMPT.format(
            question=question,
            raw_answer=pre_extracted,
        )
        try:
            normalized = await self.llm.aask(msg=prompt)
            normalized = self._clean_output(normalized)
            normalized = self._restore_specificity(question, pre_extracted, normalized)
            if normalized:
                return normalized
        except Exception as e:
            logger.warning(
                f"AnswerNormalizer LLM call failed: {e}, using rule-based fallback"
            )

        return self._clean_output(pre_extracted)

    def _rule_extract(self, question: str, raw_answer: str) -> str:
        """
        Extract obvious final-answer spans before calling the LLM.
        """
        q_lower = question.lower()

        match = re.search(
            r"(?:so\s+)?the\s+answer\s+is[:\s]+(.+?)(?:\.|$)",
            raw_answer,
            re.IGNORECASE,
        )
        if match:
            raw_answer = match.group(1).strip()

        match = re.search(r"Answer:\s*(.+?)(?:\.|$)", raw_answer)
        if match:
            raw_answer = match.group(1).strip()

        if self._asks_temporal(q_lower):
            temporal = self._extract_temporal_phrase(raw_answer)
            if temporal:
                return temporal

        if self._asks_range(q_lower):
            answer_range = self._extract_range_phrase(raw_answer)
            if answer_range:
                return answer_range

        if self._asks_ranking(q_lower):
            rank = self._extract_ranking_phrase(raw_answer)
            if rank:
                return rank

        return raw_answer

    def _clean_output(self, text: str) -> str:
        """Remove surrounding quotes, periods, extra whitespace."""
        text = text.strip().strip('"\'').strip()
        text = re.sub(r"^(?:Normalized\s+)?Answer:\s*", "", text, flags=re.IGNORECASE)
        text = text.strip()

        if text.lower().startswith("insufficient information"):
            return "Insufficient information."

        text = text.rstrip(".")
        text = self._normalize_ranking_text(text)

        # Do not split calendar dates such as "July 1, 1984".
        if "," in text and not self._has_calendar_date(text):
            tokens = [t.strip().rstrip(".").strip('"\'') for t in text.split(",")]
            tokens = [
                t for t in tokens
                if t and t.lower() not in ("etc", "etc.", "and", "or")
            ]
            if tokens:
                text = min(tokens, key=lambda t: (len(t.split()), len(t)))

        return text

    def _restore_specificity(self, question: str, raw_answer: str, normalized: str) -> str:
        """
        Keep benchmark-fragile details that the LLM normalizer often drops.
        """
        raw = self._clean_output(raw_answer)
        norm = self._clean_output(normalized)
        q_lower = question.lower()

        if not raw or raw.lower() in {"unknown", "insufficient information"}:
            return norm
        if norm.lower().startswith("insufficient information"):
            return norm

        if self._asks_temporal(q_lower):
            temporal = self._extract_temporal_phrase(raw)
            if temporal and self._is_less_specific(norm, temporal):
                return temporal

        if self._asks_range(q_lower):
            answer_range = self._extract_range_phrase(raw)
            if answer_range and self._is_less_specific(norm, answer_range):
                return answer_range

        if self._asks_ranking(q_lower):
            rank = self._extract_ranking_phrase(raw) or self._normalize_ranking_text(norm)
            if rank:
                return rank

        if self._should_keep_qualified_phrase(raw, norm):
            return raw

        return norm

    @staticmethod
    def _asks_temporal(q_lower: str) -> bool:
        return any(x in q_lower for x in (
            "when", "what date", "which date", "what year", "time period",
            "what time", "in what period", "in which period",
        ))

    @staticmethod
    def _asks_range(q_lower: str) -> bool:
        return any(x in q_lower for x in (
            "how old", "age", "between", "range", "from what age", "what ages",
        ))

    @staticmethod
    def _asks_ranking(q_lower: str) -> bool:
        return any(x in q_lower for x in (
            "ranking", "rank", "largest", "smallest", "biggest", "size",
            "oldest", "youngest", "highest", "lowest",
        ))

    @staticmethod
    def _has_calendar_date(text: str) -> bool:
        month = (
            r"January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|"
            r"Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
        )
        return bool(re.search(
            rf"\b(?:{month})\s+\d{{1,2}}(?:,|\s+of)?\s+\d{{4}}\b",
            text,
            re.IGNORECASE,
        ))

    @classmethod
    def _extract_temporal_phrase(cls, text: str) -> str:
        month = (
            r"January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|"
            r"Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
        )
        patterns = [
            rf"\b(?:{month})\s+\d{{1,2}}\s+(?:and|to|-)\s+\d{{1,2}}\s+(?:of\s+)?\d{{4}}\b",
            rf"\b(?:{month})\s+\d{{1,2}}(?:,|\s+of)?\s+\d{{4}}\b",
            r"\b(?:early|mid|late)\s+\d{1,2}(?:st|nd|rd|th)\s+(?:and|to)\s+(?:early|mid|late)\s+\d{1,2}(?:st|nd|rd|th)\s+centur(?:y|ies)\b",
            r"\b(?:early|mid|late)-?\d{1,2}(?:st|nd|rd|th)\s+to\s+(?:early|mid|late)-?\d{1,2}(?:st|nd|rd|th)\s+centur(?:y|ies)\b",
            r"\b(?:early|mid|late)\s+\d{1,2}(?:st|nd|rd|th)\s+centur(?:y|ies)\b",
            r"\b\d{4}\s*(?:-|to|and)\s*\d{4}\b",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return ""

    @staticmethod
    def _extract_range_phrase(text: str) -> str:
        patterns = [
            r"\b\d{1,3}\s*(?:-|to|and)\s*\d{1,3}\s+years?\s+old\b",
            r"\b\d{1,3}\s*(?:-|to|and)\s*\d{1,3}\b",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                value = match.group(0).strip()
                value = re.sub(r"\s*(?:-|to|and)\s*", " - ", value, flags=re.IGNORECASE)
                return value
        return ""

    @classmethod
    def _extract_ranking_phrase(cls, text: str) -> str:
        match = re.search(
            r"\b(\d+)(?:st|nd|rd|th)\s+[- ]?"
            r"(largest|smallest|biggest|oldest|youngest|highest|lowest)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return cls._ordinal_word(match.group(1)) + "-" + match.group(2).lower()

        match = re.search(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
            r"\s+[- ]?(largest|smallest|biggest|oldest|youngest|highest|lowest)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).lower() + "-" + match.group(2).lower()
        return ""

    @classmethod
    def _normalize_ranking_text(cls, text: str) -> str:
        return cls._extract_ranking_phrase(text) or text

    @staticmethod
    def _ordinal_word(num: str) -> str:
        mapping = {
            "1": "first", "2": "second", "3": "third", "4": "fourth",
            "5": "fifth", "6": "sixth", "7": "seventh", "8": "eighth",
            "9": "ninth", "10": "tenth",
        }
        return mapping.get(str(int(num)), num)

    @staticmethod
    def _is_less_specific(normalized: str, raw_specific: str) -> bool:
        norm_l = normalized.lower()
        raw_l = raw_specific.lower()
        if norm_l == raw_l:
            return False
        if norm_l in raw_l and len(raw_specific.split()) > len(normalized.split()):
            return True
        norm_nums = set(re.findall(r"\d+", normalized))
        raw_nums = set(re.findall(r"\d+", raw_specific))
        return bool(raw_nums - norm_nums)

    @staticmethod
    def _should_keep_qualified_phrase(raw: str, normalized: str) -> bool:
        if not raw or not normalized:
            return False
        if "," in raw or " and " in raw.lower():
            return False
        raw_words = raw.split()
        if len(raw_words) > 6:
            return False
        raw_l = raw.lower()
        norm_l = normalized.lower()
        if norm_l == raw_l or norm_l not in raw_l:
            return False
        qualifiers = (
            " county", " district", " province", " state", " city", " island",
            " church", " university", " college", " dollar", " pound", " euro",
            " ministry", " department", " assembly", " plant", " airport",
        )
        return any(q in raw_l for q in qualifiers)
