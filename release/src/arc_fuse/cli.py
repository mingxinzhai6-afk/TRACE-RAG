from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleLLM, ScriptedLLM
from .models import PipelineConfig
from .pipeline import ArcFusePipeline
from .retrievers import GraphTripleRetriever, LexicalRetriever, load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone ARC-Fuse reference pipeline."
    )
    parser.add_argument("--config", required=True, help="JSON configuration path")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL path")
    parser.add_argument("--graph", required=True, help="Graph triples JSONL path")
    parser.add_argument("--questions", required=True, help="Questions JSONL path")
    parser.add_argument("--output", default="output/results.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the deterministic bundled test double instead of an API.",
    )
    return parser


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = PipelineConfig.from_dict(json.load(handle))

    corpus = load_jsonl(args.corpus)
    graph = load_jsonl(args.graph)
    questions = load_jsonl(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]

    llm = ScriptedLLM() if args.offline else OpenAICompatibleLLM.from_environment()
    pipeline = ArcFusePipeline(
        llm=llm,
        graph_retriever=GraphTripleRetriever(graph),
        text_retriever=LexicalRetriever(corpus),
        config=config,
    )

    records: list[dict[str, Any]] = []
    for question_record in questions:
        question = str(question_record["question"])
        result = await pipeline.query(question)
        record = {
            "id": question_record.get("id"),
            "question": question,
            "gold_answer": question_record.get("answer"),
            **result.to_dict(),
        }
        gold = str(question_record.get("answer", "")).strip().lower()
        predicted = result.answer.strip().lower()
        record["exact_match"] = bool(gold) and gold == predicted
        records.append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    args = build_parser().parse_args()
    records = asyncio.run(run(args))
    exact = sum(bool(record["exact_match"]) for record in records)
    print(
        json.dumps(
            {
                "questions": len(records),
                "exact_match": exact,
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
