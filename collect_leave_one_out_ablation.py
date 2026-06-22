from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from generate_newg_leave_one_out_configs import VARIANT_LABELS, VARIANT_ORDER


METRICS = ["accuracy", "em", "precision", "recall", "f1"]


@dataclass(frozen=True)
class AblationRow:
    variant: str
    label: str
    n: int | None
    metrics: dict[str, float | None]
    source: Path | None


def model_short(model: str) -> str:
    short = model.split("/")[-1].split(":")[0]
    short = re.sub(r"(?i)-instruct.*", "", short)
    return re.sub(r"[^\w\-.]", "_", short).strip("_")


def score_file_candidates(results_dir: Path) -> list[Path]:
    return [
        results_dir / "results.score.json",
        results_dir / "results.score.jsonl",
        results_dir / "results.json",
        results_dir / "results.jsonl",
    ]


def result_dir_candidates(
    root: Path,
    dataset: str,
    graph: str,
    text: str,
    model: str,
    variant: str,
) -> list[Path]:
    """Return preferred result dirs for a leave-one-out variant.

    Some Full-minus-one variants are exactly equivalent to previously completed
    horizontal ablations. Prefer a fresh LOO result if present, but fall back to
    the old equivalent directory so we do not rerun expensive experiments.
    """

    base = root / "output" / dataset
    prefix = f"NewG_{graph}_{text}_{model_short(model)}"
    loo = base / f"{prefix}_loo_{variant}"

    if variant == "full":
        return [
            loo,
            base / prefix,
            base / f"{prefix}_abl_normalizer",
        ]
    if variant == "no_commendor":
        return [
            loo,
            base / f"{prefix}_abl_no_commendor",
        ]
    if variant == "no_normalizer":
        return [
            loo,
            base / f"{prefix}_abl_critic",
        ]
    if variant == "single_agent":
        return [
            loo,
            base / f"{prefix}_abl_single_agent",
        ]
    return [loo]


def find_score_file(
    root: Path,
    dataset: str,
    graph: str,
    text: str,
    model: str,
    variant: str,
) -> Path | None:
    for result_dir in result_dir_candidates(root, dataset, graph, text, model, variant):
        for candidate in score_file_candidates(result_dir / "Results"):
            if candidate.exists():
                return candidate
    return None


def load_records(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        return [parsed]

    rows = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed_line = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON line: {exc}") from exc
        if isinstance(parsed_line, dict):
            rows.append(parsed_line)
    return rows


def average_metrics(records: Iterable[dict]) -> dict[str, float | None]:
    rows = list(records)
    values: dict[str, float | None] = {}
    for metric in METRICS:
        nums = []
        for row in rows:
            value = row.get(metric)
            if value is None or value == "":
                continue
            try:
                nums.append(float(value))
            except (TypeError, ValueError):
                continue
        values[metric] = None if not nums else sum(nums) / len(nums) * 100
    return values


def collect_rows(
    root: Path,
    dataset: str,
    graph: str,
    text: str,
    model: str,
    variants: list[str],
) -> list[AblationRow]:
    rows = []
    for variant in variants:
        path = find_score_file(root, dataset, graph, text, model, variant)
        if path is None:
            rows.append(
                AblationRow(
                    variant=variant,
                    label=VARIANT_LABELS[variant],
                    n=None,
                    metrics={metric: None for metric in METRICS},
                    source=None,
                )
            )
            continue
        records = load_records(path)
        rows.append(
            AblationRow(
                variant=variant,
                label=VARIANT_LABELS[variant],
                n=len(records),
                metrics=average_metrics(records),
                source=path,
            )
        )
    return rows


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def write_outputs(rows: list[AblationRow], out_prefix: Path) -> tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv = Path(f"{out_prefix}.tsv")
    md = Path(f"{out_prefix}.md")

    header = ["variant", "label", "n", *METRICS, "source"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.variant,
                    row.label,
                    "" if row.n is None else str(row.n),
                    *[fmt(row.metrics[metric]) for metric in METRICS],
                    "" if row.source is None else str(row.source),
                ]
            )
        )
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_lines = [
        "| Variant | n | Accuracy | EM | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row.label,
                    "" if row.n is None else str(row.n),
                    *[fmt(row.metrics[metric]) for metric in METRICS],
                ]
            )
            + " |"
        )
    md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return tsv, md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect NewG leave-one-out ablation results.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", default="datasets/Popqa")
    parser.add_argument("--graph", default="hipporag")
    parser.add_argument("--text", default="bm25")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--variants", nargs="+", default=VARIANT_ORDER, choices=VARIANT_ORDER)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(
        root=args.root,
        dataset=args.dataset,
        graph=args.graph,
        text=args.text,
        model=args.model,
        variants=args.variants,
    )
    out_prefix = (
        args.out_dir
        / f"leave_one_out_{Path(args.dataset).name}_{model_short(args.model)}_{args.graph}_{args.text}"
    )
    tsv, md = write_outputs(rows, out_prefix)
    print(tsv)
    print(md)
    for row in rows:
        values = "\t".join(fmt(row.metrics[metric]) for metric in METRICS)
        print(f"{row.variant}\t{row.n or ''}\t{values}")


if __name__ == "__main__":
    main()
