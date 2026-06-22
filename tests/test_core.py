from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from arc_fuse.cli import run
from arc_fuse.fusion import rrf_fuse
from arc_fuse.models import CriticFeedback
from arc_fuse.pipeline import Critic


ROOT = Path(__file__).resolve().parents[1]


class FusionTests(unittest.TestCase):
    def test_rrf_merges_duplicate_ids_and_sources(self) -> None:
        fused = rrf_fuse(
            [{"id": "x", "content": "graph", "rank": 0}],
            [{"id": "x", "content": "text", "rank": 0}],
            top_k=5,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["sources"], ["graph", "text"])
        self.assertGreater(fused[0]["rrf_score"], 0)

    def test_actionable_critic_pass_is_not_accepted(self) -> None:
        feedback = CriticFeedback(
            verdict="pass",
            missing_entities=["France"],
        )
        guarded = Critic._guard_actionable_pass(feedback)
        self.assertEqual(guarded.verdict, "retrieve_more")


class OfflinePipelineTests(unittest.TestCase):
    def test_synthetic_demo_answers_all_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "results.jsonl"

            class Args:
                config = str(ROOT / "configs" / "arc_fuse.example.json")
                corpus = str(ROOT / "examples" / "corpus.jsonl")
                graph = str(ROOT / "examples" / "graph.jsonl")
                questions = str(ROOT / "examples" / "questions.jsonl")
                limit = 0
                offline = True

            Args.output = str(output)
            records = asyncio.run(run(Args()))

            self.assertEqual(len(records), 3)
            self.assertTrue(all(record["exact_match"] for record in records))
            persisted = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(persisted), 3)


if __name__ == "__main__":
    unittest.main()
