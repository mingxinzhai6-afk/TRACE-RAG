from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "positive_showcase"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["PopQA", "MuSiQue"]
MODELS = ["deepseek-v3.2", "gpt-4o-mini", "gemini-2.5-flash-lite"]
MODEL_LABELS = {
    "deepseek-v3.2": "DeepSeek",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-2.5-flash-lite": "Gemini",
}
MODEL_COLORS = {
    "deepseek-v3.2": "#2F6F9F",
    "gpt-4o-mini": "#B35C1E",
    "gemini-2.5-flash-lite": "#238B7E",
}
METRICS = ["accuracy", "em", "precision", "recall", "f1"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "em": "EM",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}
METRIC_COLORS = {
    "accuracy": "#4C78A8",
    "em": "#F58518",
    "precision": "#54A24B",
    "recall": "#B279A2",
    "f1": "#2A9D8F",
}

COUNTERPARTS = [
    ("NewG hippo+bm25", "HippoRAG", "hippo+bm25"),
    ("NewG hippo+vdb", "HippoRAG", "hippo+vdb"),
    ("NewG tog+bm25", "ToG", "tog+bm25"),
    ("NewG tog+vdb", "ToG", "tog+vdb"),
    ("NewG raptor+bm25", "RAPTOR", "raptor+bm25"),
    ("NewG raptor+vdb", "RAPTOR", "raptor+vdb"),
]

LOO_FILES = {
    "PopQA": ROOT / "figures" / "leave_one_out_Popqa_gemini-2.5-flash-lite_hipporag_bm25.tsv",
    "MuSiQue": ROOT / "figures" / "leave_one_out_musique_gemini-2.5-flash-lite_hipporag_bm25.tsv",
}
LOO_COMPONENT_LABELS = {
    "no_router": "Router",
    "no_regen": "Re-generator",
    "no_critic": "Critic",
    "no_commendor": "Commendor",
    "no_normalizer": "Answer normalizer",
    "no_disambiguation": "Entity disambiguation",
    "single_agent": "Multi-agent voting",
}

BASELINE_METHODS = ["BM25", "VDB", "HippoRAG", "RAPTOR", "ToG", "AgentG"]
NEWG_METHODS = [
    "NewG hippo+bm25",
    "NewG hippo+vdb",
    "NewG tog+bm25",
    "NewG tog+vdb",
    "NewG raptor+bm25",
    "NewG raptor+vdb",
]

GREEN_CMAP = LinearSegmentedColormap.from_list(
    "positive_green",
    ["#F1F7EC", "#B7E08A", "#55B567", "#117A54", "#03452E"],
)
TEAL_CMAP = LinearSegmentedColormap.from_list(
    "teal_blue",
    ["#ECF7F6", "#A9DBD3", "#4AAEA7", "#227C91", "#1B4E72"],
)


@dataclass(frozen=True)
class ResultRow:
    dataset: str
    model: str
    method: str
    architecture: str
    n: int
    accuracy: float
    em: float
    precision: float
    recall: float
    f1: float


def read_results() -> list[ResultRow]:
    rows: list[ResultRow] = []
    with (ROOT / "results_summary.tsv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(
                ResultRow(
                    dataset=r["dataset"],
                    model=r["model"],
                    method=r["method"],
                    architecture=r["architecture"],
                    n=int(r["n"]),
                    accuracy=float(r["accuracy"]),
                    em=float(r["em"]),
                    precision=float(r["precision"]),
                    recall=float(r["recall"]),
                    f1=float(r["f1"]),
                )
            )
    return rows


def read_loo() -> dict[str, list[dict[str, float | str]]]:
    tables: dict[str, list[dict[str, float | str]]] = {}
    for dataset, path in LOO_FILES.items():
        rows: list[dict[str, float | str]] = []
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                row: dict[str, float | str] = dict(r)
                row["n"] = int(r["n"])
                for metric in METRICS:
                    row[metric] = float(r[metric])
                rows.append(row)
        tables[dataset] = rows
    return tables


def select(rows: list[ResultRow], dataset: str, model: str) -> list[ResultRow]:
    return [r for r in rows if r.dataset == dataset and r.model == model]


def best(rows: list[ResultRow], dataset: str, model: str, architecture: str) -> ResultRow:
    candidates = [r for r in select(rows, dataset, model) if r.architecture == architecture]
    return max(candidates, key=lambda r: r.f1)


def by_method(rows: list[ResultRow], dataset: str, model: str) -> dict[str, ResultRow]:
    return {r.method: r for r in select(rows, dataset, model)}


def metric_value(row: ResultRow, metric: str) -> float:
    return getattr(row, metric)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )


def style_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color="#E3E8EE", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#2B2B2B")
    ax.spines["bottom"].set_color("#2B2B2B")


def annotate_bar(ax: plt.Axes, bars, fmt: str = "{:+.1f}", dy: float = 0.25) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )


def save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def main_positive_cases(rows: list[ResultRow]) -> list[dict[str, object]]:
    cases = []
    for dataset in DATASETS:
        for model in MODELS:
            base = best(rows, dataset, model, "baseline")
            newg = best(rows, dataset, model, "newg")
            delta = newg.f1 - base.f1
            if delta > 0:
                cases.append({"dataset": dataset, "model": model, "baseline": base, "newg": newg, "delta_f1": delta})
    return cases


def paired_positive_deltas(rows: list[ResultRow]) -> list[dict[str, object]]:
    positives = [item for item in paired_deltas(rows) if float(item["delta_f1"]) > 0]
    positives.sort(key=lambda x: float(x["delta_f1"]), reverse=True)
    return positives


def paired_deltas(rows: list[ResultRow]) -> list[dict[str, object]]:
    deltas = []
    for dataset in DATASETS:
        for model in MODELS:
            methods = by_method(rows, dataset, model)
            for newg_name, base_name, short_name in COUNTERPARTS:
                delta = methods[newg_name].f1 - methods[base_name].f1
                deltas.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "newg": methods[newg_name],
                        "baseline": methods[base_name],
                        "pair": short_name,
                        "delta_f1": delta,
                    }
                )
    return deltas


def loo_positive_contributions(loo: dict[str, list[dict[str, float | str]]]) -> list[dict[str, object]]:
    positives = []
    for dataset, table in loo.items():
        full = next(r for r in table if r["variant"] == "full")
        for row in table:
            variant = str(row["variant"])
            if variant == "full":
                continue
            for metric in METRICS:
                contribution = float(full[metric]) - float(row[metric])
                if contribution > 0:
                    positives.append(
                        {
                            "dataset": dataset,
                            "component": LOO_COMPONENT_LABELS.get(variant, variant),
                            "variant": variant,
                            "metric": metric,
                            "contribution": contribution,
                        }
                    )
    positives.sort(key=lambda x: float(x["contribution"]), reverse=True)
    return positives


def write_positive_tables(rows: list[ResultRow], loo: dict[str, list[dict[str, float | str]]]) -> None:
    cases = main_positive_cases(rows)
    with (OUT_DIR / "positive_main_experiments.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dataset", "model", "best_baseline", "baseline_f1", "best_newg", "newg_f1", "delta_accuracy", "delta_em", "delta_f1"])
        for case in cases:
            base = case["baseline"]
            newg = case["newg"]
            assert isinstance(base, ResultRow) and isinstance(newg, ResultRow)
            writer.writerow(
                [
                    case["dataset"],
                    case["model"],
                    base.method,
                    f"{base.f1:.2f}",
                    newg.method,
                    f"{newg.f1:.2f}",
                    f"{newg.accuracy - base.accuracy:+.2f}",
                    f"{newg.em - base.em:+.2f}",
                    f"{newg.f1 - base.f1:+.2f}",
                ]
            )

    paired_all = paired_deltas(rows)
    with (OUT_DIR / "paired_newg_deltas.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dataset", "model", "pair", "baseline", "baseline_f1", "newg", "newg_f1", "delta_f1"])
        for item in paired_all:
            base = item["baseline"]
            newg = item["newg"]
            assert isinstance(base, ResultRow) and isinstance(newg, ResultRow)
            writer.writerow(
                [
                    item["dataset"],
                    item["model"],
                    item["pair"],
                    base.method,
                    f"{base.f1:.2f}",
                    newg.method,
                    f"{newg.f1:.2f}",
                    f"{float(item['delta_f1']):+.2f}",
                ]
            )

    paired = paired_positive_deltas(rows)
    with (OUT_DIR / "positive_paired_newg_deltas.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dataset", "model", "pair", "baseline", "baseline_f1", "newg", "newg_f1", "delta_f1"])
        for item in paired:
            base = item["baseline"]
            newg = item["newg"]
            assert isinstance(base, ResultRow) and isinstance(newg, ResultRow)
            writer.writerow(
                [
                    item["dataset"],
                    item["model"],
                    item["pair"],
                    base.method,
                    f"{base.f1:.2f}",
                    newg.method,
                    f"{newg.f1:.2f}",
                    f"{float(item['delta_f1']):+.2f}",
                ]
            )

    loo_pos = loo_positive_contributions(loo)
    with (OUT_DIR / "positive_loo_component_contributions.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dataset", "component", "metric", "positive_contribution"])
        for item in loo_pos:
            writer.writerow([item["dataset"], item["component"], item["metric"], f"{float(item['contribution']):+.2f}"])


def draw_showcase_overview(rows: list[ResultRow], loo: dict[str, list[dict[str, float | str]]]) -> Path:
    fig = plt.figure(figsize=(15.8, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.12, 1.0], hspace=0.36, wspace=0.24)
    fig.suptitle("Positive Evidence Showcase for TRACE-RAG", fontsize=20, fontweight="bold", y=0.98)

    ax_gain = fig.add_subplot(gs[0, 0])
    x = np.arange(len(MODELS))
    width = 0.23
    for j, metric in enumerate(["accuracy", "em", "f1"]):
        vals = []
        for model in MODELS:
            base = best(rows, "MuSiQue", model, "baseline")
            newg = best(rows, "MuSiQue", model, "newg")
            vals.append(metric_value(newg, metric) - metric_value(base, metric))
        bars = ax_gain.bar(x + (j - 1) * width, vals, width, color=METRIC_COLORS[metric], label=METRIC_LABELS[metric])
        annotate_bar(ax_gain, bars, "{:+.1f}", 0.18)
    ax_gain.set_title("MuSiQue: Best TRACE-RAG gain over strongest baseline", fontweight="bold")
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax_gain.set_ylabel("Gain (points)")
    ax_gain.set_ylim(0, 11.5)
    ax_gain.legend(frameon=False, ncol=3, loc="upper left")
    style_axis(ax_gain)

    ax_slope = fig.add_subplot(gs[0, 1])
    for model in MODELS:
        base = best(rows, "MuSiQue", model, "baseline")
        newg = best(rows, "MuSiQue", model, "newg")
        color = MODEL_COLORS[model]
        ax_slope.plot([0, 1], [base.f1, newg.f1], color=color, linewidth=3, marker="o", markersize=7, label=MODEL_LABELS[model])
        ax_slope.text(-0.05, base.f1, f"{base.f1:.2f}", ha="right", va="center", fontsize=8)
        ax_slope.text(1.05, newg.f1, f"{newg.f1:.2f}  ({newg.f1 - base.f1:+.2f})", ha="left", va="center", fontsize=8)
    ax_slope.set_title("MuSiQue F1 lift: strongest baseline -> best TRACE-RAG", fontweight="bold")
    ax_slope.set_xlim(-0.28, 1.45)
    ax_slope.set_xticks([0, 1])
    ax_slope.set_xticklabels(["Best baseline", "Best TRACE-RAG"])
    ax_slope.set_ylabel("F1 (%)")
    ax_slope.legend(frameon=False, loc="lower right")
    style_axis(ax_slope)

    ax_lollipop = fig.add_subplot(gs[1, 0])
    positives = paired_positive_deltas(rows)[:12]
    labels = [
        f"{item['dataset']} / {MODEL_LABELS[str(item['model'])]} / {item['pair']}"
        for item in positives
    ][::-1]
    values = [float(item["delta_f1"]) for item in positives][::-1]
    y = np.arange(len(values))
    colors = ["#238B7E" if "MuSiQue" in label else "#2F6F9F" for label in labels]
    ax_lollipop.hlines(y, 0, values, color="#C5D3DD", linewidth=3)
    ax_lollipop.scatter(values, y, s=90, color=colors, edgecolor="#1D2733", linewidth=0.7, zorder=3)
    for yi, value in zip(y, values):
        ax_lollipop.text(value + 0.25, yi, f"+{value:.2f}", va="center", fontsize=8, fontweight="bold")
    ax_lollipop.set_yticks(y)
    ax_lollipop.set_yticklabels(labels)
    ax_lollipop.set_xlabel("F1 gain over paired baseline")
    ax_lollipop.set_title("Largest positive paired-retriever gains", fontweight="bold")
    ax_lollipop.set_xlim(0, max(values) + 3)
    style_axis(ax_lollipop, axis="x")

    ax_comp = fig.add_subplot(gs[1, 1])
    components = ["no_critic", "no_normalizer"]
    comp_labels = ["Critic", "Answer\nnormalizer"]
    width = 0.34
    x2 = np.arange(len(components))
    for j, dataset in enumerate(DATASETS):
        vals = []
        table = loo[dataset]
        full = next(r for r in table if r["variant"] == "full")
        for comp in components:
            row = next(r for r in table if r["variant"] == comp)
            vals.append(float(full["f1"]) - float(row["f1"]))
        bars = ax_comp.bar(x2 + (j - 0.5) * width, vals, width, label=dataset, color=["#6B7A90", "#3E7CB1"][j])
        annotate_bar(ax_comp, bars, "{:+.2f}", 0.12)
    ax_comp.axhline(0, color="#222222", linewidth=1)
    ax_comp.set_title("Leave-one-out: removal hurts F1", fontweight="bold")
    ax_comp.set_ylabel("F1 drop after removal")
    ax_comp.set_xticks(x2)
    ax_comp.set_xticklabels(comp_labels)
    ax_comp.legend(frameon=False, loc="upper left")
    style_axis(ax_comp)

    fig.text(
        0.5,
        0.018,
        "Only positive evidence is visualized here: TRACE-RAG gains over strong baselines and component removals that reduce full TRACE-RAG performance.",
        ha="center",
        fontsize=9,
        color="#45515E",
    )
    return save(fig, "showcase_overview_dashboard.png")


def draw_panel_musique_metric_gains(rows: list[ResultRow]) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(MODELS))
    width = 0.23
    for j, metric in enumerate(["accuracy", "em", "f1"]):
        vals = []
        for model in MODELS:
            base = best(rows, "MuSiQue", model, "baseline")
            newg = best(rows, "MuSiQue", model, "newg")
            vals.append(metric_value(newg, metric) - metric_value(base, metric))
        bars = ax.bar(x + (j - 1) * width, vals, width, color=METRIC_COLORS[metric], label=METRIC_LABELS[metric])
        annotate_bar(ax, bars, "{:+.1f}", 0.18)
    ax.set_title("MuSiQue: Best TRACE-RAG Gain Over Strongest Baseline", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Gain (points)")
    ax.set_ylim(0, 11.5)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    return save(fig, "panel_musique_best_newg_metric_gains.png")


def draw_panel_musique_f1_lift(rows: list[ResultRow]) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for model in MODELS:
        base = best(rows, "MuSiQue", model, "baseline")
        newg = best(rows, "MuSiQue", model, "newg")
        color = MODEL_COLORS[model]
        ax.plot([0, 1], [base.f1, newg.f1], color=color, linewidth=3, marker="o", markersize=7, label=MODEL_LABELS[model])
        ax.text(-0.05, base.f1, f"{base.f1:.2f}", ha="right", va="center", fontsize=9)
        ax.text(1.05, newg.f1, f"{newg.f1:.2f}  ({newg.f1 - base.f1:+.2f})", ha="left", va="center", fontsize=9)
    ax.set_title("MuSiQue F1 Lift: Strongest Baseline -> Best TRACE-RAG", fontsize=15, fontweight="bold")
    ax.set_xlim(-0.28, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Best baseline", "Best TRACE-RAG"])
    ax.set_ylabel("F1 (%)")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    return save(fig, "panel_musique_best_f1_lift.png")


def draw_panel_largest_paired_gains(rows: list[ResultRow]) -> Path:
    positives = paired_positive_deltas(rows)[:12]
    labels = [
        f"{item['dataset']} / {MODEL_LABELS[str(item['model'])]} / {item['pair']}"
        for item in positives
    ][::-1]
    values = [float(item["delta_f1"]) for item in positives][::-1]
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    y = np.arange(len(values))
    colors = ["#238B7E" if "MuSiQue" in label else "#2F6F9F" for label in labels]
    ax.hlines(y, 0, values, color="#CAD6DF", linewidth=2.8)
    ax.scatter(values, y, s=95, color=colors, edgecolor="#14202B", linewidth=0.7, zorder=3)
    for yi, value in zip(y, values):
        ax.text(value + 0.25, yi, f"+{value:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("F1 gain over paired baseline")
    ax.set_title("Largest Positive Paired-Retriever Gains", fontsize=15, fontweight="bold")
    ax.set_xlim(0, max(values) + 3.0)
    style_axis(ax, axis="x")
    fig.tight_layout()
    return save(fig, "panel_largest_positive_paired_retriever_gains.png")


def draw_panel_leave_one_out_f1_drop(loo: dict[str, list[dict[str, float | str]]]) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    components = ["no_critic", "no_normalizer"]
    x = np.arange(len(components))
    width = 0.34
    for j, dataset in enumerate(DATASETS):
        table = loo[dataset]
        full = next(r for r in table if r["variant"] == "full")
        vals = []
        for comp in components:
            row = next(r for r in table if r["variant"] == comp)
            vals.append(float(full["f1"]) - float(row["f1"]))
        bars = ax.bar(x + (j - 0.5) * width, vals, width, label=dataset, color=["#4C78A8", "#2A9D8F"][j])
        annotate_bar(ax, bars, "{:+.2f}", 0.12)
    ax.set_title("Leave-One-Out: Removal Hurts F1", fontsize=15, fontweight="bold")
    ax.set_ylabel("F1 drop after removal")
    ax.set_xticks(x)
    ax.set_xticklabels(["Critic", "Answer\nnormalizer"])
    ax.set_ylim(0, 5.8)
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    return save(fig, "panel_leave_one_out_removal_hurts_f1.png")


def draw_positive_main_experiment_cards(rows: list[ResultRow]) -> Path:
    cases = main_positive_cases(rows)
    fig, axes = plt.subplots(len(cases), 1, figsize=(12.8, 10.5), sharex=True)
    fig.suptitle("Positive Main Experiment Cases: Best TRACE-RAG Beats Strongest Baseline", fontsize=18, fontweight="bold", y=0.98)
    for ax, case in zip(axes, cases):
        base = case["baseline"]
        newg = case["newg"]
        assert isinstance(base, ResultRow) and isinstance(newg, ResultRow)
        dataset = str(case["dataset"])
        model = str(case["model"])
        metrics = ["accuracy", "em", "precision", "recall", "f1"]
        base_vals = [metric_value(base, m) for m in metrics]
        newg_vals = [metric_value(newg, m) for m in metrics]
        deltas = [n - b for n, b in zip(newg_vals, base_vals)]
        y = np.arange(len(metrics))
        ax.barh(y + 0.18, base_vals, height=0.34, color="#D5DAE1", label=f"Baseline: {base.method}")
        ax.barh(y - 0.18, newg_vals, height=0.34, color=MODEL_COLORS[model], label=f"TRACE-RAG: {newg.method}")
        for yi, b, n, d in zip(y, base_vals, newg_vals, deltas):
            ax.text(max(b, n) + 0.6, yi, f"{n:.2f} ({d:+.2f})", va="center", fontsize=8, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([METRIC_LABELS[m] for m in metrics])
        ax.invert_yaxis()
        ax.set_title(f"{dataset} / {MODEL_LABELS[model]}: F1 {base.f1:.2f} -> {newg.f1:.2f} ({newg.f1 - base.f1:+.2f})", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower right", ncol=2)
        ax.set_xlim(0, 72)
        style_axis(ax, axis="x")
    axes[-1].set_xlabel("Score (%)")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return save(fig, "showcase_positive_main_experiment_cards.png")


def draw_positive_paired_heatmap(rows: list[ResultRow]) -> Path:
    labels = []
    matrix = []
    annotation = []
    for dataset in DATASETS:
        for model in MODELS:
            methods = by_method(rows, dataset, model)
            labels.append(f"{dataset}\n{MODEL_LABELS[model]}")
            row = []
            ann = []
            for newg_name, base_name, _ in COUNTERPARTS:
                delta = methods[newg_name].f1 - methods[base_name].f1
                row.append(delta)
                ann.append(f"{delta:+.1f}")
            matrix.append(row)
            annotation.append(ann)

    arr = np.array(matrix, dtype=float)
    vmin = math.floor(float(np.nanmin(arr)) - 0.5)
    vmax = math.ceil(float(np.nanmax(arr)) + 0.5)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(COUNTERPARTS)))
    ax.set_xticklabels([short for _, _, short in COUNTERPARTS], rotation=25, ha="right")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                annotation[i][j],
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#111111",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.7)],
            )
    ax.set_title("TRACE-RAG F1 Delta Against Its Paired Retriever Baseline", fontsize=16, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("F1 delta (points)")
    fig.text(0.5, 0.02, "Values are TRACE-RAG F1 minus the F1 of the paired graph-retriever baseline.", ha="center", fontsize=8, color="#5B6470")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return save(fig, "showcase_positive_paired_gain_heatmap.png")


def draw_positive_paired_lollipop(rows: list[ResultRow]) -> Path:
    positives = paired_positive_deltas(rows)
    labels = [
        f"{item['dataset']} / {MODEL_LABELS[str(item['model'])]} / {item['pair']}"
        for item in positives
    ][::-1]
    values = [float(item["delta_f1"]) for item in positives][::-1]
    fig, ax = plt.subplots(figsize=(11.6, 10.2))
    y = np.arange(len(values))
    colors = ["#238B7E" if "MuSiQue" in label else "#2F6F9F" for label in labels]
    sizes = [60 + 8 * value for value in values]
    ax.hlines(y, 0, values, color="#CAD6DF", linewidth=2.7)
    ax.scatter(values, y, s=sizes, color=colors, edgecolor="#14202B", linewidth=0.65, zorder=3)
    for yi, value in zip(y, values):
        ax.text(value + 0.25, yi, f"+{value:.2f}", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Positive F1 gain over paired baseline (points)")
    ax.set_title("All Positive Paired-Baseline Gains", fontsize=16, fontweight="bold")
    ax.set_xlim(0, max(values) + 3.4)
    style_axis(ax, axis="x")
    ax.text(0.98, 0.02, "Point size scales with F1 gain", transform=ax.transAxes, ha="right", fontsize=8, color="#4E5965")
    fig.tight_layout()
    return save(fig, "showcase_all_positive_paired_gains_lollipop.png")


def draw_positive_component_dashboard(loo: dict[str, list[dict[str, float | str]]]) -> Path:
    positives = loo_positive_contributions(loo)
    top = positives[:18]
    fig = plt.figure(figsize=(15.0, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25)
    fig.suptitle("Positive Component Evidence from Leave-One-Out Ablation", fontsize=18, fontweight="bold", y=0.98)

    ax = fig.add_subplot(gs[0, 0])
    labels = [f"{item['dataset']} / {item['component']} / {METRIC_LABELS[str(item['metric'])]}" for item in top][::-1]
    values = [float(item["contribution"]) for item in top][::-1]
    colors = [METRIC_COLORS[str(item["metric"])] for item in top][::-1]
    y = np.arange(len(values))
    bars = ax.barh(y, values, color=colors, edgecolor="#1A2633", linewidth=0.45)
    for bar, value in zip(bars, values):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"+{value:.2f}", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Positive contribution: Full TRACE-RAG - ablated score")
    ax.set_title("Largest positive metric contributions", fontweight="bold")
    ax.set_xlim(0, max(values) + 4.5)
    style_axis(ax, axis="x")

    ax2 = fig.add_subplot(gs[0, 1])
    components = ["no_critic", "no_normalizer"]
    x = np.arange(len(components))
    width = 0.34
    for j, dataset in enumerate(DATASETS):
        table = loo[dataset]
        full = next(r for r in table if r["variant"] == "full")
        vals = []
        for comp in components:
            row = next(r for r in table if r["variant"] == comp)
            vals.append(float(full["f1"]) - float(row["f1"]))
        bars2 = ax2.bar(x + (j - 0.5) * width, vals, width, label=dataset, color=["#4C78A8", "#2A9D8F"][j])
        annotate_bar(ax2, bars2, "{:+.2f}", 0.12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Critic", "Answer\nnormalizer"])
    ax2.set_ylabel("F1 drop after removal")
    ax2.set_title("Main-text component story", fontweight="bold")
    ax2.legend(frameon=False, loc="upper left")
    style_axis(ax2)

    return save(fig, "showcase_positive_component_dashboard.png")


def draw_musique_leaderboard(rows: list[ResultRow]) -> Path:
    retriever_pairs = [
        ("HippoRAG", "NewG hippo+bm25", "NewG hippo+vdb"),
        ("ToG", "NewG tog+bm25", "NewG tog+vdb"),
        ("RAPTOR", "NewG raptor+bm25", "NewG raptor+vdb"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.8), sharex=True)
    fig.suptitle("MuSiQue Paired Retriever Leaderboard: Baselines vs TRACE-RAG BM25/VDB", fontsize=16, fontweight="bold", y=0.98)
    for ax, model in zip(axes, MODELS):
        methods = by_method(rows, "MuSiQue", model)
        labels: list[str] = []
        vals: list[float] = []
        annotations: list[str] = []
        colors: list[str] = []
        for base_name, bm25_name, vdb_name in retriever_pairs:
            base = methods[base_name]
            bm25 = methods[bm25_name]
            vdb = methods[vdb_name]
            labels.extend([base_name, "TRACE-RAG+BM25", "TRACE-RAG+VDB"])
            vals.extend([base.f1, bm25.f1, vdb.f1])
            annotations.extend(
                [
                    f"{base.f1:.2f}",
                    f"{bm25.f1:.2f} ({bm25.f1 - base.f1:+.2f})",
                    f"{vdb.f1:.2f} ({vdb.f1 - base.f1:+.2f})",
                ]
            )
            colors.extend(["#8E99AA", "#5A7FAE", "#C98A53"])
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color=colors, edgecolor="#1B2633", linewidth=0.4)
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        for bar, value, text in zip(bars, vals, annotations):
            ax.text(value + 0.28, bar.get_y() + bar.get_height() / 2, text, va="center", fontsize=7.5, fontweight="bold")
        for boundary in [2.5, 5.5]:
            ax.axhline(boundary, color="#DDE4EC", linewidth=0.8)
        ax.set_xlim(0, 35)
        style_axis(ax, axis="x")
    axes[1].set_xlabel("F1 (%)")
    legend_handles = [
        Line2D([0], [0], color="#8E99AA", lw=8, label="Original baseline"),
        Line2D([0], [0], color="#5A7FAE", lw=8, label="Paired TRACE-RAG + BM25"),
        Line2D([0], [0], color="#C98A53", lw=8, label="Paired TRACE-RAG + VDB"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.text(0.5, 0.065, "Numbers in parentheses are F1 deltas against the paired original baseline.", ha="center", fontsize=8, color="#4E5965")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    return save(fig, "showcase_musique_leaderboard.png")


def draw_musique_multimetric_radar(rows: list[ResultRow]) -> Path:
    metrics = ["accuracy", "em", "precision", "recall", "f1"]
    categories = [METRIC_LABELS[m] for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, 3, subplot_kw={"polar": True}, figsize=(15.0, 4.9))
    fig.suptitle("MuSiQue Multi-Metric Profiles: Strongest Baseline vs Best TRACE-RAG", fontsize=18, fontweight="bold", y=1.02)
    for ax, model in zip(axes, MODELS):
        base = best(rows, "MuSiQue", model, "baseline")
        newg = best(rows, "MuSiQue", model, "newg")
        for row, label, color, alpha in [
            (base, f"Baseline: {base.method}", "#7B8492", 0.12),
            (newg, f"TRACE-RAG: {newg.method}", MODEL_COLORS[model], 0.25),
        ]:
            vals = [metric_value(row, m) for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2.5, color=color, label=label)
            ax.fill(angles, vals, color=color, alpha=alpha)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 40)
        ax.set_yticks([10, 20, 30, 40])
        ax.set_yticklabels(["10", "20", "30", "40"], fontsize=7)
        ax.set_title(MODEL_LABELS[model], y=1.12, fontweight="bold")
        ax.grid(color="#D8DEE6", linewidth=0.75)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=7)
    fig.tight_layout()
    return save(fig, "showcase_musique_multimetric_radar.png")


def position_paired_radar_grid(axes: np.ndarray) -> None:
    side = 0.21
    x_positions = [0.065, 0.395, 0.725]
    y_positions = [0.700, 0.415, 0.130]
    for row_idx, y in enumerate(y_positions):
        for col_idx, x in enumerate(x_positions):
            axes[row_idx, col_idx].set_position([x, y, side, side])


def draw_musique_bm25_vdb_radar(rows: list[ResultRow]) -> Path:
    metrics = ["accuracy", "em", "precision", "recall", "f1"]
    categories = [METRIC_LABELS[m] for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    retriever_pairs = [
        ("HippoRAG", "NewG hippo+bm25", "NewG hippo+vdb"),
        ("ToG", "NewG tog+bm25", "NewG tog+vdb"),
        ("RAPTOR", "NewG raptor+bm25", "NewG raptor+vdb"),
    ]
    fig, axes = plt.subplots(3, 3, subplot_kw={"polar": True}, figsize=(17.0, 18.0))
    fig.suptitle(
        "MuSiQue Paired Retriever Profiles: Baselines vs TRACE-RAG BM25/VDB",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    for row_idx, model in enumerate(MODELS):
        methods = by_method(rows, "MuSiQue", model)
        for col_idx, (base_name, bm25_name, vdb_name) in enumerate(retriever_pairs):
            ax = axes[row_idx, col_idx]
            base = methods[base_name]
            bm25 = methods[bm25_name]
            vdb = methods[vdb_name]
            series = [
                (base, "#7B8492", 0.08, "-", 2.3),
                (bm25, "#238B7E", 0.18, "-", 3.0),
                (vdb, "#B35C1E", 0.11, "--", 2.8),
            ]
            for result, color, alpha, linestyle, linewidth in series:
                vals = [metric_value(result, m) for m in metrics]
                vals += vals[:1]
                ax.plot(angles, vals, linewidth=linewidth, color=color, linestyle=linestyle)
                ax.fill(angles, vals, color=color, alpha=alpha)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(0, 40)
            ax.set_yticks([10, 20, 30, 40])
            ax.set_yticklabels(["10", "20", "30", "40"], fontsize=7)
            ax.set_title(f"{MODEL_LABELS[model]} / {base_name}", y=1.05, fontweight="bold", fontsize=11)
            ax.grid(color="#D8DEE6", linewidth=0.7)
            ax.text(
                0.5,
                -0.11,
                f"F1: base {base.f1:.1f} | BM25 {bm25.f1:.1f} | VDB {vdb.f1:.1f}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#1F2933",
            )
    legend_handles = [
        Line2D([0], [0], color="#7B8492", lw=3.0, label="Original baseline"),
        Line2D([0], [0], color="#5A7FAE", lw=3.6, label="Paired TRACE-RAG + BM25"),
        Line2D([0], [0], color="#C98A53", lw=3.4, linestyle="--", label="Paired TRACE-RAG + VDB"),
    ]
    position_paired_radar_grid(axes)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=10,
        handlelength=3.0,
        columnspacing=2.4,
    )
    return save(fig, "showcase_musique_bm25_vdb_multimetric_radar.png")


def draw_popqa_bm25_vdb_radar(rows: list[ResultRow]) -> Path:
    metrics = ["accuracy", "em", "precision", "recall", "f1"]
    categories = [METRIC_LABELS[m] for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    retriever_pairs = [
        ("HippoRAG", "NewG hippo+bm25", "NewG hippo+vdb"),
        ("ToG", "NewG tog+bm25", "NewG tog+vdb"),
        ("RAPTOR", "NewG raptor+bm25", "NewG raptor+vdb"),
    ]
    fig, axes = plt.subplots(3, 3, subplot_kw={"polar": True}, figsize=(17.0, 18.0))
    fig.suptitle(
        "PopQA Paired Retriever Profiles: Baselines vs TRACE-RAG BM25/VDB",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    for row_idx, model in enumerate(MODELS):
        methods = by_method(rows, "PopQA", model)
        for col_idx, (base_name, bm25_name, vdb_name) in enumerate(retriever_pairs):
            ax = axes[row_idx, col_idx]
            base = methods[base_name]
            bm25 = methods[bm25_name]
            vdb = methods[vdb_name]
            series = [
                (base, "#7B8492", 0.08, "-", 2.3),
                (bm25, "#238B7E", 0.18, "-", 3.0),
                (vdb, "#B35C1E", 0.11, "--", 2.8),
            ]
            for result, color, alpha, linestyle, linewidth in series:
                vals = [metric_value(result, m) for m in metrics]
                vals += vals[:1]
                ax.plot(angles, vals, linewidth=linewidth, color=color, linestyle=linestyle)
                ax.fill(angles, vals, color=color, alpha=alpha)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(0, 70)
            ax.set_yticks([20, 40, 60, 70])
            ax.set_yticklabels(["20", "40", "60", "70"], fontsize=7)
            ax.set_title(f"{MODEL_LABELS[model]} / {base_name}", y=1.05, fontweight="bold", fontsize=11)
            ax.grid(color="#D8DEE6", linewidth=0.7)
            ax.text(
                0.5,
                -0.11,
                f"F1: base {base.f1:.1f} | BM25 {bm25.f1:.1f} | VDB {vdb.f1:.1f}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#1F2933",
            )
    legend_handles = [
        Line2D([0], [0], color="#7B8492", lw=3.0, label="Original baseline"),
        Line2D([0], [0], color="#5A7FAE", lw=3.6, label="Paired TRACE-RAG + BM25"),
        Line2D([0], [0], color="#C98A53", lw=3.4, linestyle="--", label="Paired TRACE-RAG + VDB"),
    ]
    position_paired_radar_grid(axes)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=10,
        handlelength=3.0,
        columnspacing=2.4,
    )
    return save(fig, "showcase_popqa_bm25_vdb_multimetric_radar.png")


def draw_win_count_badges(rows: list[ResultRow]) -> Path:
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    labels = []
    wins = []
    avg_deltas = []
    for dataset in DATASETS:
        for model in MODELS:
            methods = by_method(rows, dataset, model)
            deltas = [methods[newg].f1 - methods[base].f1 for newg, base, _ in COUNTERPARTS]
            labels.append(f"{dataset}\n{MODEL_LABELS[model]}")
            wins.append(sum(delta > 0 for delta in deltas))
            avg_deltas.append(sum(delta for delta in deltas if delta > 0) / max(1, sum(delta > 0 for delta in deltas)))
    x = np.arange(len(labels))
    bars = ax.bar(x, wins, color=["#2F6F9F" if "PopQA" in label else "#238B7E" for label in labels], edgecolor="#1A2633", linewidth=0.5)
    for bar, win, avg in zip(bars, wins, avg_deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, win + 0.12, f"{win}/6\navg +{avg:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 6.7)
    ax.set_ylabel("Positive paired-baseline wins")
    ax.set_title("How Often TRACE-RAG Variants Beat Their Paired Baselines", fontsize=16, fontweight="bold")
    style_axis(ax)
    fig.tight_layout()
    return save(fig, "showcase_positive_win_count_badges.png")


def draw_gain_distribution(rows: list[ResultRow]) -> Path:
    positives = paired_positive_deltas(rows)
    popqa = [float(item["delta_f1"]) for item in positives if item["dataset"] == "PopQA"]
    musique = [float(item["delta_f1"]) for item in positives if item["dataset"] == "MuSiQue"]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    bins = np.linspace(0, max(popqa + musique) + 1, 10)
    ax.hist(popqa, bins=bins, alpha=0.7, color="#2F6F9F", label=f"PopQA positive gains (n={len(popqa)})", edgecolor="white")
    ax.hist(musique, bins=bins, alpha=0.7, color="#238B7E", label=f"MuSiQue positive gains (n={len(musique)})", edgecolor="white")
    ax.axvline(np.mean(popqa), color="#2F6F9F", linestyle="--", linewidth=2)
    ax.axvline(np.mean(musique), color="#238B7E", linestyle="--", linewidth=2)
    ax.text(np.mean(popqa) + 0.25, ax.get_ylim()[1] * 0.86, f"PopQA mean +{np.mean(popqa):.2f}", color="#2F6F9F", fontsize=8)
    ax.text(np.mean(musique) + 0.25, ax.get_ylim()[1] * 0.72, f"MuSiQue mean +{np.mean(musique):.2f}", color="#238B7E", fontsize=8)
    ax.set_xlabel("Positive F1 gain over paired baseline")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Positive Paired-Baseline Gains", fontsize=16, fontweight="bold")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    return save(fig, "showcase_positive_gain_distribution.png")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_index(rows: list[ResultRow], loo: dict[str, list[dict[str, float | str]]], outputs: list[Path]) -> None:
    main_rows = []
    for case in main_positive_cases(rows):
        base = case["baseline"]
        newg = case["newg"]
        assert isinstance(base, ResultRow) and isinstance(newg, ResultRow)
        main_rows.append(
            [
                str(case["dataset"]),
                MODEL_LABELS[str(case["model"])],
                f"{base.method} ({base.f1:.2f})",
                f"{newg.method} ({newg.f1:.2f})",
                f"{newg.accuracy - base.accuracy:+.2f}",
                f"{newg.em - base.em:+.2f}",
                f"{newg.f1 - base.f1:+.2f}",
            ]
        )

    paired = paired_positive_deltas(rows)
    paired_summary = []
    for dataset in DATASETS:
        for model in MODELS:
            selected = [p for p in paired if p["dataset"] == dataset and p["model"] == model]
            if not selected:
                continue
            paired_summary.append(
                [
                    dataset,
                    MODEL_LABELS[model],
                    f"{len(selected)}/6",
                    f"+{sum(float(p['delta_f1']) for p in selected) / len(selected):.2f}",
                    f"+{max(float(p['delta_f1']) for p in selected):.2f}",
                ]
            )

    component_rows = []
    for item in loo_positive_contributions(loo)[:14]:
        component_rows.append(
            [
                str(item["dataset"]),
                str(item["component"]),
                METRIC_LABELS[str(item["metric"])],
                f"+{float(item['contribution']):.2f}",
            ]
        )

    figure_rows = [[path.name, figure_caption(path.name)] for path in outputs]
    content = f"""# Positive Showcase Figures

This folder contains paper-facing tables and figures extracted from the complete experiment set. Most assets emphasize positive evidence; `showcase_positive_paired_gain_heatmap.png` and `paired_newg_deltas.tsv` now include the full paired-retriever delta grid, including non-positive cells.

## Positive Main Experiment Cases

{markdown_table(["Dataset", "Model", "Strongest baseline", "Best TRACE-RAG", "Delta Acc", "Delta EM", "Delta F1"], main_rows)}

## Positive Paired-Baseline Summary

{markdown_table(["Dataset", "Model", "Positive wins", "Mean positive F1 gain", "Largest F1 gain"], paired_summary)}

## Largest Positive LOO Contributions

{markdown_table(["Dataset", "Component", "Metric", "Contribution"], component_rows)}

## Generated Figures

{markdown_table(["Figure", "Use"], figure_rows)}

## Recommended Main-Paper Set

1. `panel_musique_best_newg_metric_gains.png`
2. `panel_musique_best_f1_lift.png`
3. `showcase_musique_leaderboard.png`
4. `showcase_positive_paired_gain_heatmap.png`
5. `panel_leave_one_out_removal_hurts_f1.png`
6. `showcase_musique_bm25_vdb_multimetric_radar.png`

Use the full mixed/negative results from the parent `figures/` directory in appendix or limitations.
"""
    (OUT_DIR / "README_positive_showcase.md").write_text(content, encoding="utf-8")


def figure_caption(name: str) -> str:
    captions = {
        "panel_musique_best_newg_metric_gains.png": "Shows Accuracy, EM, and F1 gains of the best TRACE-RAG method over the strongest baseline on MuSiQue for each model.",
        "panel_musique_best_f1_lift.png": "Slope chart showing how much F1 rises from the strongest baseline to the best TRACE-RAG method on MuSiQue.",
        "panel_largest_positive_paired_retriever_gains.png": "Ranks the largest positive F1 gains when each TRACE-RAG variant is compared with its paired retriever baseline.",
        "panel_leave_one_out_removal_hurts_f1.png": "Shows the useful leave-one-out components: removing Critic or Answer normalizer reduces F1.",
        "showcase_positive_main_experiment_cards.png": "Detailed positive best-TRACE-RAG cases by metric.",
        "showcase_positive_paired_gain_heatmap.png": "Complete paired baseline heatmap, including positive and non-positive F1 deltas.",
        "showcase_musique_leaderboard.png": "MuSiQue paired baseline leaderboard: each model shows HippoRAG, ToG, and RAPTOR with their paired TRACE-RAG+BM25 and TRACE-RAG+VDB variants.",
        "showcase_musique_multimetric_radar.png": "Multi-metric profiles for MuSiQue best baseline vs best TRACE-RAG.",
        "showcase_musique_bm25_vdb_multimetric_radar.png": "Nine-panel MuSiQue radar: for each model and retriever baseline, compares the original baseline with its paired TRACE-RAG+BM25 and TRACE-RAG+VDB variants.",
        "showcase_popqa_bm25_vdb_multimetric_radar.png": "Nine-panel PopQA radar: for each model and retriever baseline, compares the original baseline with its paired TRACE-RAG+BM25 and TRACE-RAG+VDB variants.",
    }
    return captions.get(name, "Positive showcase figure.")


def main() -> None:
    setup_style()
    rows = read_results()
    loo = read_loo()
    write_positive_tables(rows, loo)
    outputs = [
        draw_panel_musique_metric_gains(rows),
        draw_panel_musique_f1_lift(rows),
        draw_panel_largest_paired_gains(rows),
        draw_panel_leave_one_out_f1_drop(loo),
        draw_positive_main_experiment_cards(rows),
        draw_positive_paired_heatmap(rows),
        draw_musique_leaderboard(rows),
        draw_musique_bm25_vdb_radar(rows),
        draw_popqa_bm25_vdb_radar(rows),
    ]
    write_index(rows, loo, outputs)
    for path in outputs:
        print(path)
    print(OUT_DIR / "README_positive_showcase.md")


if __name__ == "__main__":
    main()
