from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODELS = ["deepseek-v3.2", "gpt-4o-mini", "gemini-2.5-flash-lite"]

DATASET_DIRS = {
    "PopQA": "Popqa",
    "MuSiQue": "musique",
}

METHOD_SPECS = [
    ("BM25", "BM25", "baseline"),
    ("VDB", "VDB", "baseline"),
    ("HippoRAG", "HippoRAG", "baseline"),
    ("RAPTOR", "RAPTOR", "baseline"),
    ("ToG", "ToG", "baseline"),
    ("AgentG", "AgentG", "baseline"),
    ("NewG_hipporag_bm25", "NewG hippo+bm25", "newg"),
    ("NewG_hipporag_vdb", "NewG hippo+vdb", "newg"),
    ("NewG_tog_bm25", "NewG tog+bm25", "newg"),
    ("NewG_tog_vdb", "NewG tog+vdb", "newg"),
    ("NewG_raptor_bm25", "NewG raptor+bm25", "newg"),
    ("NewG_raptor_vdb", "NewG raptor+vdb", "newg"),
]

METRIC_SPECS = [
    ("accuracy", "accuracy"),
    ("em", "em"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("f1", "f1"),
]


# Working-note values from the current PopQA table. Real result files override
# this fallback whenever they are present.
STATIC_POPQA = {
    "deepseek-v3.2": {
        "BM25": {"accuracy": 65.50, "em": 65.50, "precision": 66.58, "recall": 21.95, "f1": 28.86},
        "VDB": {"accuracy": 55.50, "em": 55.50, "precision": 56.00, "recall": 17.82, "f1": 23.59},
        "HippoRAG": {"accuracy": 61.00, "em": 61.00, "precision": 61.50, "recall": 21.09, "f1": 27.34},
        "RAPTOR": {"accuracy": 41.00, "em": 40.50, "precision": 41.67, "recall": 13.76, "f1": 17.78},
        "ToG": {"accuracy": 63.00, "em": 38.00, "precision": 49.59, "recall": 23.19, "f1": 26.16},
        "AgentG": {"accuracy": 53.50, "em": 31.50, "precision": 42.43, "recall": 19.60, "f1": 22.90},
        "NewG hippo+bm25": {"accuracy": 65.50, "em": 65.00, "precision": 66.83, "recall": 23.26, "f1": 29.78},
        "NewG hippo+vdb": {"accuracy": 64.00, "em": 63.50, "precision": 64.83, "recall": 22.00, "f1": 28.39},
        "NewG tog+bm25": {"accuracy": 58.00, "em": 58.00, "precision": 58.83, "recall": 20.13, "f1": 25.96},
        "NewG tog+vdb": {"accuracy": 58.50, "em": 58.50, "precision": 59.33, "recall": 19.98, "f1": 25.80},
        "NewG raptor+bm25": {"accuracy": 64.50, "em": 64.00, "precision": 64.92, "recall": 20.78, "f1": 27.41},
        "NewG raptor+vdb": {"accuracy": 59.50, "em": 59.50, "precision": 60.75, "recall": 18.95, "f1": 25.09},
    },
    "gpt-4o-mini": {
        "BM25": {"accuracy": 65.00, "em": 64.50, "precision": 65.83, "recall": 22.17, "f1": 28.86},
        "VDB": {"accuracy": 54.00, "em": 53.50, "precision": 54.25, "recall": 17.48, "f1": 23.02},
        "HippoRAG": {"accuracy": 65.50, "em": 36.00, "precision": 48.36, "recall": 25.46, "f1": 28.24},
        "RAPTOR": {"accuracy": 40.50, "em": 39.00, "precision": 41.42, "recall": 13.28, "f1": 17.33},
        "ToG": {"accuracy": 50.50, "em": 27.00, "precision": 38.30, "recall": 19.84, "f1": 22.32},
        "AgentG": {"accuracy": 51.00, "em": 38.50, "precision": 45.63, "recall": 19.68, "f1": 23.37},
        "NewG hippo+bm25": {"accuracy": 64.00, "em": 64.00, "precision": 64.83, "recall": 22.73, "f1": 29.06},
        "NewG hippo+vdb": {"accuracy": 61.50, "em": 61.50, "precision": 62.33, "recall": 21.75, "f1": 27.95},
        "NewG tog+bm25": {"accuracy": 49.50, "em": 49.50, "precision": 50.33, "recall": 16.34, "f1": 21.43},
        "NewG tog+vdb": {"accuracy": 51.00, "em": 50.50, "precision": 51.83, "recall": 16.92, "f1": 22.16},
        "NewG raptor+bm25": {"accuracy": 58.00, "em": 58.00, "precision": 58.50, "recall": 19.64, "f1": 25.50},
        "NewG raptor+vdb": {"accuracy": 55.00, "em": 55.00, "precision": 55.50, "recall": 17.83, "f1": 23.50},
    },
    "gemini-2.5-flash-lite": {
        "BM25": {"accuracy": 62.00, "em": 58.00, "precision": 61.08, "recall": 21.73, "f1": 26.98},
        "VDB": {"accuracy": 56.00, "em": 51.50, "precision": 54.00, "recall": 19.27, "f1": 23.57},
        "HippoRAG": {"accuracy": 61.50, "em": 58.50, "precision": 60.92, "recall": 22.51, "f1": 27.85},
        "RAPTOR": {"accuracy": 38.50, "em": 32.50, "precision": 36.04, "recall": 13.83, "f1": 16.55},
        "ToG": {"accuracy": 31.00, "em": 26.00, "precision": 27.40, "recall": 9.94, "f1": 11.97},
        "AgentG": {"accuracy": 48.00, "em": 27.50, "precision": 37.92, "recall": 16.80, "f1": 19.53},
        "NewG hippo+bm25": {"accuracy": 59.50, "em": 58.50, "precision": 59.83, "recall": 21.69, "f1": 27.50},
        "NewG hippo+vdb": {"accuracy": 57.50, "em": 56.50, "precision": 57.83, "recall": 20.93, "f1": 26.57},
        "NewG tog+bm25": {"accuracy": 57.00, "em": 56.00, "precision": 57.25, "recall": 20.32, "f1": 25.97},
        "NewG tog+vdb": {"accuracy": 54.00, "em": 53.00, "precision": 54.25, "recall": 19.46, "f1": 24.71},
        "NewG raptor+bm25": {"accuracy": 58.00, "em": 57.00, "precision": 58.25, "recall": 19.37, "f1": 25.32},
        "NewG raptor+vdb": {"accuracy": 53.00, "em": 52.00, "precision": 53.25, "recall": 16.95, "f1": 22.41},
    },
}

STATIC_POPQA_NOTES = {
}


# Working-note values from MuSiQue Table 2. DeepSeek/GPT BM25 and VDB are
# intentionally absent until their reruns finish.
STATIC_MUSIQUE = {
    "deepseek-v3.2": {
        "BM25": {"accuracy": 13.50, "em": 13.00, "precision": 18.59, "recall": 16.93, "f1": 17.24},
        "VDB": {"accuracy": 8.00, "em": 8.00, "precision": 11.33, "recall": 10.33, "f1": 10.65},
        "HippoRAG": {"accuracy": 14.50, "em": 12.50, "precision": 18.94, "recall": 17.23, "f1": 17.28},
        "RAPTOR": {"accuracy": 18.50, "em": 18.50, "precision": 32.52, "recall": 26.51, "f1": 28.17},
        "ToG": {"accuracy": 20.50, "em": 18.50, "precision": 34.67, "recall": 28.93, "f1": 30.06},
        "AgentG": {"accuracy": 6.00, "em": 6.00, "precision": 12.29, "recall": 10.52, "f1": 11.04},
        "NewG hippo+bm25": {"accuracy": 19.50, "em": 17.50, "precision": 33.29, "recall": 26.77, "f1": 28.42},
        "NewG hippo+vdb": {"accuracy": 21.50, "em": 19.00, "precision": 34.96, "recall": 28.48, "f1": 30.01},
        "NewG tog+bm25": {"accuracy": 21.00, "em": 19.00, "precision": 36.54, "recall": 29.78, "f1": 31.44},
        "NewG tog+vdb": {"accuracy": 21.50, "em": 20.00, "precision": 37.25, "recall": 29.65, "f1": 31.44},
        "NewG raptor+bm25": {"accuracy": 14.00, "em": 13.50, "precision": 30.92, "recall": 22.98, "f1": 25.13},
        "NewG raptor+vdb": {"accuracy": 13.00, "em": 12.50, "precision": 28.33, "recall": 21.17, "f1": 23.07},
    },
    "gpt-4o-mini": {
        "BM25": {"accuracy": 16.50, "em": 16.00, "precision": 23.91, "recall": 21.77, "f1": 22.22},
        "VDB": {"accuracy": 10.50, "em": 9.50, "precision": 14.42, "recall": 13.33, "f1": 13.40},
        "HippoRAG": {"accuracy": 21.00, "em": 17.00, "precision": 25.62, "recall": 25.64, "f1": 24.90},
        "RAPTOR": {"accuracy": 15.00, "em": 14.50, "precision": 27.42, "recall": 22.56, "f1": 23.76},
        "ToG": {"accuracy": 14.00, "em": 13.00, "precision": 23.84, "recall": 21.43, "f1": 21.55},
        "AgentG": {"accuracy": 10.00, "em": 9.50, "precision": 14.13, "recall": 12.54, "f1": 12.98},
        "NewG hippo+bm25": {"accuracy": 23.50, "em": 22.00, "precision": 36.47, "recall": 30.39, "f1": 31.77},
        "NewG hippo+vdb": {"accuracy": 22.00, "em": 20.50, "precision": 32.62, "recall": 27.48, "f1": 28.59},
        "NewG tog+bm25": {"accuracy": 22.00, "em": 20.50, "precision": 32.72, "recall": 27.83, "f1": 28.98},
        "NewG tog+vdb": {"accuracy": 23.00, "em": 20.50, "precision": 31.68, "recall": 28.59, "f1": 29.13},
        "NewG raptor+bm25": {"accuracy": 22.50, "em": 22.00, "precision": 31.28, "recall": 28.00, "f1": 28.80},
        "NewG raptor+vdb": {"accuracy": 16.00, "em": 15.00, "precision": 22.92, "recall": 20.49, "f1": 21.09},
    },
    "gemini-2.5-flash-lite": {
        "BM25": {"accuracy": 9.00, "em": 9.00, "precision": 13.63, "recall": 12.46, "f1": 12.78},
        "VDB": {"accuracy": 4.50, "em": 4.50, "precision": 7.42, "recall": 6.83, "f1": 6.98},
        "HippoRAG": {"accuracy": 16.00, "em": 14.00, "precision": 19.32, "recall": 18.13, "f1": 18.13},
        "RAPTOR": {"accuracy": 14.00, "em": 14.00, "precision": 23.08, "recall": 20.14, "f1": 20.99},
        "ToG": {"accuracy": 5.50, "em": 4.50, "precision": 8.72, "recall": 8.36, "f1": 7.71},
        "AgentG": {"accuracy": 6.00, "em": 6.00, "precision": 11.67, "recall": 9.23, "f1": 9.92},
        "NewG hippo+bm25": {"accuracy": 19.50, "em": 18.50, "precision": 27.79, "recall": 25.95, "f1": 26.37},
        "NewG hippo+vdb": {"accuracy": 18.00, "em": 16.00, "precision": 24.91, "recall": 23.10, "f1": 23.40},
        "NewG tog+bm25": {"accuracy": 24.00, "em": 21.50, "precision": 32.09, "recall": 30.08, "f1": 30.26},
        "NewG tog+vdb": {"accuracy": 21.50, "em": 20.00, "precision": 28.37, "recall": 26.73, "f1": 26.94},
        "NewG raptor+bm25": {"accuracy": 18.50, "em": 17.50, "precision": 24.71, "recall": 22.90, "f1": 23.24},
        "NewG raptor+vdb": {"accuracy": 11.50, "em": 11.50, "precision": 17.75, "recall": 14.89, "f1": 15.77},
    },
}

STATIC_BY_DATASET = {
    "PopQA": STATIC_POPQA,
    "MuSiQue": STATIC_MUSIQUE,
}

STATIC_NOTES_BY_DATASET = {
    "PopQA": STATIC_POPQA_NOTES,
}


@dataclass(frozen=True)
class ResultRow:
    dataset: str
    model: str
    method: str
    architecture: str
    n: int | None
    metrics: dict[str, float | None]
    source: str
    notes: str = ""


def score_file_candidates(results_dir: Path) -> list[Path]:
    return [
        results_dir / "results.score.json",
        results_dir / "results.score.jsonl",
        results_dir / "results.json",
        results_dir / "results.jsonl",
    ]


def find_score_file(root: Path, dataset: str, method_key: str, model: str) -> Path | None:
    dataset_dir = DATASET_DIRS[dataset]
    results_dir = root / "output" / "datasets" / dataset_dir / f"{method_key}_{model}" / "Results"
    for candidate in score_file_candidates(results_dir):
        if candidate.exists():
            return candidate
    return None


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        return [parsed]

    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed_line = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON line: {exc}") from exc
        if isinstance(parsed_line, dict):
            records.append(parsed_line)
    return records


def average_metric(records: Iterable[dict], key: str) -> float | None:
    values = []
    for record in records:
        value = record.get(key)
        if value is None or value == "":
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    avg = sum(values) / len(values)
    return avg * 100 if abs(avg) <= 1.000001 else avg


def metrics_from_records(records: list[dict]) -> dict[str, float | None]:
    return {metric_key: average_metric(records, metric_key) for metric_key, _ in METRIC_SPECS}


def static_metrics(dataset: str, model: str, method_label: str) -> dict[str, float | None] | None:
    metrics = STATIC_BY_DATASET.get(dataset, {}).get(model, {}).get(method_label)
    if metrics is None:
        return None
    return dict(metrics)


def collect_results(
    dataset: str,
    root: Path = Path("."),
    include_static: bool = False,
    expected_n: int = 200,
    prefer_static_on_bad_n: bool = False,
    truncate_overlong: bool = False,
    prefer_static: bool = False,
) -> list[ResultRow]:
    rows = []
    root = root.resolve()
    for model in MODELS:
        for method_key, method_label, architecture in METHOD_SPECS:
            source_file = find_score_file(root, dataset, method_key, model)
            notes = []
            static = static_metrics(dataset, model, method_label) if include_static else None
            if prefer_static and static is not None:
                rows.append(
                    ResultRow(
                        dataset=dataset,
                        model=model,
                        method=method_label,
                        architecture=architecture,
                        n=expected_n,
                        metrics=static,
                        source="final_table_2026-05-16",
                        notes="user supplied final table",
                    )
                )
                continue
            if source_file is not None:
                records = load_records(source_file)
                original_n = len(records)
                if truncate_overlong and original_n > expected_n:
                    records = records[:expected_n]
                    notes.append(f"truncated_from_n={original_n}")
                n = len(records)
                metrics = metrics_from_records(records)
                source = str(source_file.relative_to(root))
                if n != expected_n:
                    notes.append(f"expected_n={expected_n}")
                    if prefer_static_on_bad_n and static is not None:
                        metrics = static
                        source = "static_working_notes"
                        notes.append(f"ignored_mismatched_file_n={n}")
                        n = expected_n
            else:
                if static is None:
                    n = None
                    source = "missing"
                    metrics = {metric_key: None for metric_key, _ in METRIC_SPECS}
                else:
                    n = expected_n
                    source = "static_working_notes"
                    metrics = static
                    notes.append("static fallback")

            static_note = STATIC_NOTES_BY_DATASET.get(dataset, {}).get((model, method_label))
            if source_file is None and static_note and include_static:
                notes.append(static_note)

            rows.append(
                ResultRow(
                    dataset=dataset,
                    model=model,
                    method=method_label,
                    architecture=architecture,
                    n=n,
                    metrics=metrics,
                    source=source,
                    notes="; ".join(notes),
                )
            )
    return rows


def collect_result_map(
    dataset: str,
    root: Path = Path("."),
    include_static: bool = False,
    expected_n: int = 200,
    prefer_static_on_bad_n: bool = False,
    truncate_overlong: bool = False,
    prefer_static: bool = False,
) -> dict[tuple[str, str], ResultRow]:
    return {
        (row.model, row.method): row
        for row in collect_results(
            dataset,
            root=root,
            include_static=include_static,
            expected_n=expected_n,
            prefer_static_on_bad_n=prefer_static_on_bad_n,
            truncate_overlong=truncate_overlong,
            prefer_static=prefer_static,
        )
    }


def fmt_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.2f}"


def render_markdown(rows: list[ResultRow]) -> str:
    headers = ["dataset", "model", "method", "arch", "n", "accuracy", "em", "precision", "recall", "f1", "source", "notes"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row.dataset,
            row.model,
            row.method,
            row.architecture,
            "" if row.n is None else str(row.n),
            *(fmt_metric(row.metrics[metric_key]) for metric_key, _ in METRIC_SPECS),
            row.source,
            row.notes,
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_tsv(rows: list[ResultRow]) -> str:
    headers = ["dataset", "model", "method", "architecture", "n", "accuracy", "em", "precision", "recall", "f1", "source", "notes"]
    lines = ["\t".join(headers)]
    for row in rows:
        values = [
            row.dataset,
            row.model,
            row.method,
            row.architecture,
            "" if row.n is None else str(row.n),
            *(fmt_metric(row.metrics[metric_key]) for metric_key, _ in METRIC_SPECS),
            row.source,
            row.notes,
        ]
        lines.append("\t".join(values))
    return "\n".join(lines)


def render_csv(rows: list[ResultRow]) -> str:
    headers = ["dataset", "model", "method", "architecture", "n", "accuracy", "em", "precision", "recall", "f1", "source", "notes"]
    out = []
    writer = csv.writer(_ListWriter(out))
    writer.writerow(headers)
    for row in rows:
        writer.writerow(
            [
                row.dataset,
                row.model,
                row.method,
                row.architecture,
                "" if row.n is None else str(row.n),
                *(fmt_metric(row.metrics[metric_key]) for metric_key, _ in METRIC_SPECS),
                row.source,
                row.notes,
            ]
        )
    return "".join(out)


class _ListWriter:
    def __init__(self, out: list[str]) -> None:
        self.out = out

    def write(self, value: str) -> None:
        self.out.append(value)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GraphRAG/NewG score files into a result table.")
    parser.add_argument("--dataset", choices=["PopQA", "MuSiQue", "all"], default="all")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--format", choices=["markdown", "tsv", "csv"], default="markdown")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-static", action="store_true", help="Fill missing PopQA cells from working-note fallback values.")
    parser.add_argument(
        "--prefer-static",
        action="store_true",
        help="Use final static table values even when local result files are present.",
    )
    parser.add_argument(
        "--prefer-static-on-bad-n",
        action="store_true",
        help="When --include-static is set, use fallback values for PopQA files whose row count is not --expected-n.",
    )
    parser.add_argument(
        "--truncate-overlong",
        action="store_true",
        help="For files with more than --expected-n rows, evaluate only the first --expected-n rows and mark the source.",
    )
    parser.add_argument("--expected-n", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    datasets = list(DATASET_DIRS) if args.dataset == "all" else [args.dataset]
    rows = []
    for dataset in datasets:
        rows.extend(
            collect_results(
                dataset,
                root=args.root,
                include_static=args.include_static,
                expected_n=args.expected_n,
                prefer_static_on_bad_n=args.prefer_static_on_bad_n,
                truncate_overlong=args.truncate_overlong,
                prefer_static=args.prefer_static,
            )
        )

    if args.format == "markdown":
        text = render_markdown(rows)
    elif args.format == "tsv":
        text = render_tsv(rows)
    else:
        text = render_csv(rows)

    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
