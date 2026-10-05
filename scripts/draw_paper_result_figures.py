from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "figures" / "generated_results"
DEFAULT_PAIRED_DATA = ROOT / "Data" / "plotting" / "paired_retriever_deltas.tsv"
OUT_B = DEFAULT_OUT
PAIRED_DATA = DEFAULT_PAIRED_DATA


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig: plt.Figure, name: str, extra_dir: Path | None = None) -> None:
    for out_dir in [extra_dir or OUT_B]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:+.2f}") -> None:
    for bar in bars:
        value = bar.get_height()
        y = value + (0.35 if value >= 0 else -0.55)
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=10.5,
        )


def draw_musique_metric_gains() -> None:
    models = ["DeepSeek", "GPT-4o-mini", "Gemini"]
    metrics = {
        "Accuracy": [0.50, 2.50, 10.00],
        "EM": [0.50, 5.00, 7.50],
        "F1": [1.38, 6.87, 9.27],
    }
    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=metric, color=colors[i])
        annotate_bars(ax, bars)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Gain")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 11.8)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    save(fig, "paper_panel_musique_metric_gains.png")


def draw_musique_f1_lift() -> None:
    models = ["DeepSeek", "GPT-4o-mini", "Gemini"]
    baseline = np.array([30.06, 24.90, 20.99])
    trace = np.array([31.44, 31.77, 30.26])
    x = np.arange(len(models))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    bars1 = ax.bar(x - width / 2, baseline, width, label="Baseline F1", color="#9ecae9")
    bars2 = ax.bar(x + width / 2, trace, width, label="TRACE-RAG F1", color="#f28e2b")
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.45,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
            )
    for i, gain in enumerate(trace - baseline):
        ax.annotate(
            f"+{gain:.2f}",
            xy=(i, trace[i]),
            xytext=(i, trace[i] + 3.0),
            ha="center",
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
            fontsize=11,
        )
    ax.set_ylabel("F1")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 38)
    ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    save(fig, "paper_panel_musique_f1_lift.png")


def draw_main_cards() -> None:
    labels = [
        "MuSiQue\nDeepSeek",
        "MuSiQue\nGPT-4o-mini",
        "MuSiQue\nGemini",
        "PopQA\nDeepSeek",
        "PopQA\nGPT-4o-mini",
        "PopQA\nGemini",
    ]
    acc = [0.50, 2.50, 10.00, 0.00, -1.00, -2.00]
    em = [0.50, 5.00, 7.50, -0.50, -0.50, 0.00]
    f1 = [1.38, 6.87, 9.27, 0.92, 0.20, -0.35]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    for i, (name, vals, color) in enumerate(
        [("Accuracy", acc, "#4c78a8"), ("EM", em, "#f58518"), ("F1", f1, "#54a24b")]
    ):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        annotate_bars(ax, bars)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Gain")
    ax.set_xticks(x, labels)
    ax.set_ylim(-3.2, 11.8)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    save(fig, "paper_showcase_main_experiment_cards.png", OUT_B)


def draw_paired_leaderboard() -> None:
    labels = [
        "Gemini\nToG",
        "Gemini\nToG+BM25",
        "Gemini\nToG+VDB",
        "DeepSeek\nHippoRAG",
        "DeepSeek\nHippo+BM25",
        "DeepSeek\nHippo+VDB",
    ]
    values = [7.71, 30.26, 26.94, 17.28, 28.42, 30.01]
    colors = ["#bdbdbd", "#f28e2b", "#59a14f", "#bdbdbd", "#f28e2b", "#59a14f"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(np.arange(len(labels)), values, color=colors)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.55,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )
    ax.set_ylabel("F1")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, 34)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    save(fig, "paper_showcase_musique_leaderboard.png")


def draw_paired_largest_gains() -> None:
    labels = [
        "MuSiQue\nGemini\ntog+bm25",
        "MuSiQue\nGemini\ntog+vdb",
        "PopQA\nGemini\ntog+bm25",
        "PopQA\nGemini\ntog+vdb",
        "MuSiQue\nDeepSeek\nhippo+vdb",
        "MuSiQue\nDeepSeek\nhippo+bm25",
    ]
    gains = [22.55, 19.23, 14.00, 12.74, 12.73, 11.14]
    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    bars = ax.bar(np.arange(len(labels)), gains, color="#4c78a8")
    annotate_bars(ax, bars)
    ax.set_ylabel("F1 Gain")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, 25.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    save(fig, "paper_panel_paired_largest_gains.png")


def draw_paired_heatmap() -> None:
    source = PAIRED_DATA
    row_keys = [
        ("MuSiQue", "deepseek-v3.2"),
        ("MuSiQue", "gpt-4o-mini"),
        ("MuSiQue", "gemini-2.5-flash-lite"),
        ("PopQA", "deepseek-v3.2"),
        ("PopQA", "gpt-4o-mini"),
        ("PopQA", "gemini-2.5-flash-lite"),
    ]
    rows = ["MuSiQue\nDeepSeek", "MuSiQue\nGPT-4o-mini", "MuSiQue\nGemini", "PopQA\nDeepSeek", "PopQA\nGPT-4o-mini", "PopQA\nGemini"]
    cols = ["hippo+bm25", "hippo+vdb", "tog+bm25", "tog+vdb", "raptor+bm25", "raptor+vdb"]
    values = {}
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            values[(row["dataset"], row["model"], row["pair"])] = float(row["delta_f1"])

    data = np.array(
        [[values[(dataset, model, pair)] for pair in cols] for dataset, model in row_keys],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(11.6, 5.9))
    im = ax.imshow(data, cmap=plt.cm.RdYlGn, vmin=-6, vmax=23)
    ax.set_xticks(np.arange(len(cols)), cols, rotation=12, ha="right")
    ax.set_yticks(np.arange(len(rows)), rows)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1 Gain")
    save(fig, "paper_showcase_paired_gain_heatmap.png", OUT_B)


def draw_loo_f1() -> None:
    labels = ["MuSiQue\nCritic", "MuSiQue\nAnswerNorm", "PopQA\nCritic", "PopQA\nAnswerNorm"]
    gains = [5.33, 3.59, 1.04, 1.45]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    bars = ax.bar(np.arange(len(labels)), gains, color=["#4c78a8", "#f28e2b", "#4c78a8", "#f28e2b"])
    annotate_bars(ax, bars)
    ax.set_ylabel("F1 Drop After Removal")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, 6.4)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    save(fig, "paper_panel_loo_f1_drop.png")


def draw_loo_top_contributions() -> None:
    entries = [
        ("PopQA AnswerNorm EM", 38.50),
        ("PopQA AnswerNorm Precision", 21.46),
        ("MuSiQue Critic Recall", 5.50),
        ("MuSiQue Critic Precision", 5.37),
        ("MuSiQue Critic F1", 5.33),
        ("MuSiQue AnswerNorm EM", 6.50),
        ("MuSiQue AnswerNorm Precision", 5.17),
        ("MuSiQue Critic Accuracy", 4.50),
        ("MuSiQue Critic EM", 4.00),
        ("MuSiQue AnswerNorm F1", 3.59),
    ]
    entries = sorted(entries, key=lambda x: x[1])
    labels = [name for name, _ in entries]
    values = [value for _, value in entries]
    fig, ax = plt.subplots(figsize=(9.6, 5.7))
    bars = ax.barh(np.arange(len(labels)), values, color="#59a14f")
    for bar in bars:
        ax.text(bar.get_width() + 0.45, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}", va="center", fontsize=10.5)
    ax.set_xlabel("Metric Drop After Removal")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlim(0, 42)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    save(fig, "paper_panel_loo_top_contributions.png")


def main() -> None:
    global OUT_B, PAIRED_DATA
    parser = argparse.ArgumentParser(
        description="Recreate the main TRACE-RAG experiment figures from paper data."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUT,
        help="Directory for generated PNG figures (default: figures/generated_results).",
    )
    parser.add_argument(
        "--paired-data", type=Path, default=DEFAULT_PAIRED_DATA,
        help="TSV containing paired graph-text F1 results for the heatmap.",
    )
    args = parser.parse_args()
    OUT_B = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    PAIRED_DATA = args.paired_data if args.paired_data.is_absolute() else ROOT / args.paired_data
    if not PAIRED_DATA.exists():
        parser.error(f"paired data file does not exist: {PAIRED_DATA}")
    draw_musique_metric_gains()
    draw_musique_f1_lift()
    draw_main_cards()
    draw_paired_leaderboard()
    draw_paired_largest_gains()
    draw_paired_heatmap()
    draw_loo_f1()
    draw_loo_top_contributions()


if __name__ == "__main__":
    main()
