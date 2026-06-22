from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collect_leave_one_out_ablation import (
    METRICS,
    collect_rows,
    model_short,
    write_outputs,
)
from generate_newg_leave_one_out_configs import VARIANT_ORDER


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "em": "EM",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

COLORS = {
    "accuracy": "#4C78A8",
    "em": "#F58518",
    "precision": "#54A24B",
    "recall": "#B279A2",
    "f1": "#E45756",
}


def dataset_label(dataset: str) -> str:
    name = Path(dataset).name.lower()
    if name == "popqa":
        return "PopQA"
    if name == "musique":
        return "MuSiQue"
    return Path(dataset).name


def style_axes(ax) -> None:
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_metric_profile(rows, out_dir: Path, stem: str, title_prefix: str) -> Path:
    present = [row for row in rows if row.n is not None]
    labels = [row.label for row in present]
    x = np.arange(len(present))

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for metric in METRICS:
        vals = [row.metrics[metric] for row in present]
        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=COLORS[metric],
            label=METRIC_LABELS[metric],
        )
        for i, value in enumerate(vals):
            if value is not None and metric in {"em", "f1"}:
                ax.text(i, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{title_prefix}: NewG Leave-One-Out Ablation Profile", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5, frameon=False)
    style_axes(ax)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.31)
    out = out_dir / f"{stem}_profile.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def draw_delta_vs_full(rows, out_dir: Path, stem: str, title_prefix: str) -> Path:
    full = next((row for row in rows if row.variant == "full" and row.n is not None), None)
    present = [row for row in rows if row.variant != "full" and row.n is not None]
    if full is None or not present:
        raise ValueError("Need a completed full row and at least one completed ablation row to plot deltas")

    labels = [row.label for row in present]
    y = np.arange(len(present))
    height = 0.14

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.axvline(0, color="#333333", linewidth=1.0)
    for idx, metric in enumerate(METRICS):
        base = full.metrics[metric]
        deltas = [
            np.nan if base is None or row.metrics[metric] is None else row.metrics[metric] - base
            for row in present
        ]
        ax.barh(
            y + (idx - 2) * height,
            deltas,
            height,
            label=METRIC_LABELS[metric],
            color=COLORS[metric],
            edgecolor="#333333",
            linewidth=0.35,
        )

    ax.set_title(f"{title_prefix}: NewG Leave-One-Out Delta vs Full", fontsize=15, fontweight="bold")
    ax.set_xlabel("Score change from Full NewG (percentage points)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.24, right=0.98, top=0.9, bottom=0.22)
    out = out_dir / f"{stem}_delta_vs_full.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NewG leave-one-out ablation results.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", default="datasets/Popqa")
    parser.add_argument("--graph", default="hipporag")
    parser.add_argument("--text", default="bm25")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--variants", nargs="+", default=VARIANT_ORDER, choices=VARIANT_ORDER)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures") / "ablation_experiments" / "leave_one_out_ablation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = dataset_label(args.dataset)
    stem = f"leave_one_out_{Path(args.dataset).name}_{model_short(args.model)}_{args.graph}_{args.text}"
    rows = collect_rows(
        root=args.root,
        dataset=args.dataset,
        graph=args.graph,
        text=args.text,
        model=args.model,
        variants=args.variants,
    )
    write_outputs(rows, args.out_dir / stem)

    outputs = []
    if any(row.n is not None for row in rows):
        outputs.append(draw_metric_profile(rows, args.out_dir, stem, title_prefix))
    if any(row.variant == "full" and row.n is not None for row in rows):
        completed_ablation = any(row.variant != "full" and row.n is not None for row in rows)
        if completed_ablation:
            outputs.append(draw_delta_vs_full(rows, args.out_dir, stem, title_prefix))

    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
