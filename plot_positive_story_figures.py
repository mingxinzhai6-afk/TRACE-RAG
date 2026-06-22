from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
MAIN_FIG_DIR = FIG_DIR / "main_experiments"
LEAVE_ONE_OUT_FIG_DIR = FIG_DIR / "ablation_experiments" / "leave_one_out_ablation"
FIG_DIR.mkdir(exist_ok=True)
MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
LEAVE_ONE_OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ROOT / "experiment_results_organized.md"

DATASETS = ["PopQA", "MuSiQue"]
MODELS = ["deepseek-v3.2", "gpt-4o-mini", "gemini-2.5-flash-lite"]
MODEL_LABELS = {
    "deepseek-v3.2": "DeepSeek",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-2.5-flash-lite": "Gemini",
}
METRICS = ["accuracy", "em", "precision", "recall", "f1"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "em": "EM",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

COUNTERPARTS = [
    ("NewG hippo+bm25", "HippoRAG", "hippo+bm25"),
    ("NewG hippo+vdb", "HippoRAG", "hippo+vdb"),
    ("NewG tog+bm25", "ToG", "tog+bm25"),
    ("NewG tog+vdb", "ToG", "tog+vdb"),
    ("NewG raptor+bm25", "RAPTOR", "raptor+bm25"),
    ("NewG raptor+vdb", "RAPTOR", "raptor+vdb"),
]

OLD_ABLATION_METHODS = [
    "Simple",
    "+Router",
    "+Re-Generator",
    "+Critic & Commendor",
    "w/o Commendor",
    "Single judge/voter",
    "Full NewG",
]
OLD_ABLATION = {
    "accuracy": [66.00, 66.00, 66.00, 66.50, 58.50, 60.00, 59.50],
    "em": [24.00, 24.00, 19.50, 20.00, 57.50, 58.50, 58.50],
    "precision": [42.16, 42.16, 37.78, 38.38, 58.83, 60.08, 59.83],
    "recall": [26.49, 26.49, 25.78, 25.88, 21.57, 21.94, 21.69],
    "f1": [27.15, 27.15, 25.92, 26.05, 27.28, 27.81, 27.50],
}

LOO_FILES = {
    "PopQA": LEAVE_ONE_OUT_FIG_DIR / "leave_one_out_Popqa_gemini-2.5-flash-lite_hipporag_bm25.tsv",
    "MuSiQue": LEAVE_ONE_OUT_FIG_DIR / "leave_one_out_musique_gemini-2.5-flash-lite_hipporag_bm25.tsv",
}
LOO_COMPONENT_LABELS = {
    "no_router": "Router",
    "no_regen": "Re-generator",
    "no_critic": "Critic",
    "no_commendor": "Commendor",
    "no_normalizer": "Normalizer",
    "no_disambiguation": "Disambiguation",
    "single_agent": "Multi-agent voting",
}


@dataclass(frozen=True)
class MainRow:
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


def read_main_rows() -> list[MainRow]:
    rows: list[MainRow] = []
    with (ROOT / "results_summary.tsv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(
                MainRow(
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


def read_loo_tables() -> dict[str, list[dict[str, float | str]]]:
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


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#E2E6EA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def rows_for(rows: list[MainRow], dataset: str, model: str) -> list[MainRow]:
    return [r for r in rows if r.dataset == dataset and r.model == model]


def best_by_arch(rows: list[MainRow], dataset: str, model: str, architecture: str) -> MainRow:
    subset = [r for r in rows_for(rows, dataset, model) if r.architecture == architecture]
    return max(subset, key=lambda r: r.f1)


def method_map(rows: list[MainRow], dataset: str, model: str) -> dict[str, MainRow]:
    return {r.method: r for r in rows_for(rows, dataset, model)}


def row_metric(row: MainRow, metric: str) -> float:
    return getattr(row, metric)


def save(fig: plt.Figure, name: str) -> Path:
    if name.startswith("positive_loo_") or name.startswith("positive_leave_one_out_"):
        out = LEAVE_ONE_OUT_FIG_DIR / name
    else:
        out = MAIN_FIG_DIR / name
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def draw_best_newg_vs_baseline_f1(rows: list[MainRow]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    colors = ["#6B7280", "#2A9D8F"]
    for ax, dataset in zip(axes, DATASETS):
        x = np.arange(len(MODELS))
        width = 0.34
        base_vals = []
        newg_vals = []
        labels = []
        for model in MODELS:
            base = best_by_arch(rows, dataset, model, "baseline")
            newg = best_by_arch(rows, dataset, model, "newg")
            base_vals.append(base.f1)
            newg_vals.append(newg.f1)
            labels.append(MODEL_LABELS[model])
        ax.bar(x - width / 2, base_vals, width, color=colors[0], label="Best baseline")
        ax.bar(x + width / 2, newg_vals, width, color=colors[1], label="Best NewG")
        for i, (b, n) in enumerate(zip(base_vals, newg_vals)):
            delta = n - b
            color = "#0F7B5F" if delta >= 0 else "#A94442"
            ax.text(
                i,
                max(b, n) + 0.9,
                f"{delta:+.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("F1 (%)")
        ax.set_ylim(0, 36)
        style_axes(ax)
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Best NewG vs Best Baseline by Dataset and Model", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "positive_best_newg_vs_best_baseline_f1.png")


def draw_musique_metric_gains(rows: list[MainRow]) -> Path:
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    x = np.arange(len(MODELS))
    width = 0.24
    metrics = ["accuracy", "em", "f1"]
    colors = ["#4C78A8", "#F58518", "#2A9D8F"]
    for j, metric in enumerate(metrics):
        vals = []
        for model in MODELS:
            base = best_by_arch(rows, "MuSiQue", model, "baseline")
            newg = best_by_arch(rows, "MuSiQue", model, "newg")
            vals.append(row_metric(newg, metric) - row_metric(base, metric))
        bars = ax.bar(x + (j - 1) * width, vals, width, label=f"Delta {METRIC_LABELS[metric]}", color=colors[j])
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.25,
                f"{value:+.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Gain over best baseline (points)")
    ax.set_title("MuSiQue: Best NewG Gains Over Strongest Baseline", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    return save(fig, "positive_musique_best_newg_metric_gains.png")


def draw_best_gain_matrix(rows: list[MainRow]) -> Path:
    labels = []
    matrix = []
    for dataset in DATASETS:
        for model in MODELS:
            base = best_by_arch(rows, dataset, model, "baseline")
            newg = best_by_arch(rows, dataset, model, "newg")
            labels.append(f"{dataset}\n{MODEL_LABELS[model]}")
            matrix.append([newg.accuracy - base.accuracy, newg.em - base.em, newg.f1 - base.f1])
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-3.0, vmax=10.0, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Accuracy", "EM", "F1"])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9, color="#111111")
    ax.set_title("Best NewG Gain Matrix vs Strongest Baseline", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Score gain (points)")
    fig.tight_layout()
    return save(fig, "positive_best_newg_gain_matrix.png")


def draw_counterpart_delta_heatmap(rows: list[MainRow]) -> Path:
    labels = []
    matrix = []
    for dataset in DATASETS:
        for model in MODELS:
            by_method = method_map(rows, dataset, model)
            labels.append(f"{dataset}\n{MODEL_LABELS[model]}")
            matrix.append([by_method[newg].f1 - by_method[base].f1 for newg, base, _ in COUNTERPARTS])
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-6.0, vmax=14.0, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(COUNTERPARTS)))
    ax.set_xticklabels([short for _, _, short in COUNTERPARTS], rotation=25, ha="right")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8, color="#111111")
    ax.set_title("NewG F1 Delta Against Its Paired Retriever Baseline", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("F1 gain (points)")
    fig.tight_layout()
    return save(fig, "positive_counterpart_f1_delta_heatmap.png")


def draw_counterpart_win_rate(rows: list[MainRow]) -> Path:
    labels = []
    wins = []
    avg = []
    for dataset in DATASETS:
        for model in MODELS:
            by_method = method_map(rows, dataset, model)
            deltas = [by_method[newg].f1 - by_method[base].f1 for newg, base, _ in COUNTERPARTS]
            labels.append(f"{dataset} / {MODEL_LABELS[model]}")
            wins.append(sum(d > 0 for d in deltas))
            avg.append(sum(deltas) / len(deltas))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    colors = ["#2A9D8F" if value >= 4 else "#C9A227" for value in wins]
    bars = ax.barh(y, wins, color=colors, edgecolor="#333333", linewidth=0.5)
    for bar, win, mean_delta in zip(bars, wins, avg):
        ax.text(
            win + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{win}/6, avg {mean_delta:+.2f}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 6.8)
    ax.set_xlabel("Number of NewG variants beating paired baseline")
    ax.set_title("Counterpart Win Count Across NewG Variants", fontsize=14, fontweight="bold")
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return save(fig, "positive_counterpart_win_count.png")


def draw_musique_method_heatmap(rows: list[MainRow]) -> Path:
    methods = [
        "BM25",
        "VDB",
        "HippoRAG",
        "RAPTOR",
        "ToG",
        "AgentG",
        "NewG hippo+bm25",
        "NewG hippo+vdb",
        "NewG tog+bm25",
        "NewG tog+vdb",
        "NewG raptor+bm25",
        "NewG raptor+vdb",
    ]
    short = ["BM25", "VDB", "Hippo", "RAPTOR", "ToG", "AgentG", "N h+b", "N h+v", "N t+b", "N t+v", "N r+b", "N r+v"]
    matrix = []
    for model in MODELS:
        by_method = method_map(rows, "MuSiQue", model)
        matrix.append([by_method[m].f1 for m in methods])
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(12.6, 3.6))
    im = ax.imshow(arr, cmap="YlGnBu", vmin=6, vmax=33, aspect="auto")
    ax.axvline(5.5, color="#222222", linewidth=1.1)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(short, rotation=30, ha="right")
    for i in range(arr.shape[0]):
        best_j = int(np.argmax(arr[i]))
        for j in range(arr.shape[1]):
            weight = "bold" if j == best_j else "normal"
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=8, fontweight=weight)
    ax.text(2.5, -0.82, "Baselines", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(8.5, -0.82, "NewG variants", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title("MuSiQue F1 Landscape: Baselines vs NewG Variants", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("F1 (%)")
    fig.tight_layout()
    return save(fig, "positive_musique_method_f1_heatmap.png")


def loo_contributions(loo_tables: dict[str, list[dict[str, float | str]]], metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dataset, rows in loo_tables.items():
        full = next(r for r in rows if r["variant"] == "full")
        out[dataset] = {}
        for row in rows:
            variant = str(row["variant"])
            if variant == "full":
                continue
            out[dataset][variant] = float(full[metric]) - float(row[metric])
    return out


def draw_loo_component_contribution(loo_tables: dict[str, list[dict[str, float | str]]]) -> Path:
    contrib = loo_contributions(loo_tables, "f1")
    variants = ["no_router", "no_regen", "no_critic", "no_commendor", "no_normalizer", "no_disambiguation", "single_agent"]
    y = np.arange(len(variants))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    dataset_colors = {"PopQA": "#4C78A8", "MuSiQue": "#2A9D8F"}
    for j, dataset in enumerate(DATASETS):
        vals = [contrib[dataset][v] for v in variants]
        bars = ax.barh(
            y + (j - 0.5) * width,
            vals,
            width,
            color=dataset_colors[dataset],
            edgecolor="#333333",
            linewidth=0.35,
        )
        for bar, value in zip(bars, vals):
            ha = "left" if value >= 0 else "right"
            offset = 0.08 if value >= 0 else -0.08
            ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center", ha=ha, fontsize=8)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LOO_COMPONENT_LABELS[v] for v in variants])
    ax.set_xlabel("Full NewG F1 - ablated F1 (positive means component helps)")
    ax.set_title("Leave-One-Out Component Contribution", fontsize=14, fontweight="bold")
    legend_handles = [Patch(facecolor=dataset_colors[d], edgecolor="#333333", label=d) for d in DATASETS]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return save(fig, "positive_leave_one_out_component_contribution.png")


def draw_loo_key_component_metric_drop(loo_tables: dict[str, list[dict[str, float | str]]]) -> Path:
    components = ["no_critic", "no_normalizer"]
    metrics = ["accuracy", "em", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    colors = ["#4C78A8", "#F58518", "#2A9D8F"]
    for ax, dataset in zip(axes, DATASETS):
        rows = loo_tables[dataset]
        full = next(r for r in rows if r["variant"] == "full")
        x = np.arange(len(components))
        width = 0.24
        for j, metric in enumerate(metrics):
            vals = []
            for comp in components:
                ablated = next(r for r in rows if r["variant"] == comp)
                vals.append(float(full[metric]) - float(ablated[metric]))
            bars = ax.bar(x + (j - 1) * width, vals, width, label=METRIC_LABELS[metric], color=colors[j])
            for bar, value in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:+.1f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="#222222", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["Critic", "Normalizer"])
        ax.set_title(dataset)
        ax.set_ylabel("Score drop after removal (points)")
        style_axes(ax)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Key Components: Removal Hurts Full NewG", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save(fig, "positive_loo_critic_normalizer_metric_drop.png")


def draw_loo_key_component_f1_drop(loo_tables: dict[str, list[dict[str, float | str]]]) -> Path:
    components = ["no_critic", "no_normalizer"]
    x = np.arange(len(DATASETS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    colors = ["#2A9D8F", "#4C78A8"]
    for j, comp in enumerate(components):
        vals = []
        for dataset in DATASETS:
            rows = loo_tables[dataset]
            full = next(r for r in rows if r["variant"] == "full")
            ablated = next(r for r in rows if r["variant"] == comp)
            vals.append(float(full["f1"]) - float(ablated["f1"]))
        bars = ax.bar(x + (j - 0.5) * width, vals, width, color=colors[j], label=LOO_COMPONENT_LABELS[comp])
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:+.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("F1 drop after component removal (points)")
    ax.set_title("Key Leave-One-Out Drops in F1", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    return save(fig, "positive_loo_key_component_f1_drop.png")


def draw_musique_slope(rows: list[MainRow]) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = [0, 1]
    for idx, model in enumerate(MODELS):
        base = best_by_arch(rows, "MuSiQue", model, "baseline")
        newg = best_by_arch(rows, "MuSiQue", model, "newg")
        y = [base.f1, newg.f1]
        color = ["#4C78A8", "#E45756", "#2A9D8F"][idx]
        ax.plot(x, y, marker="o", linewidth=2.5, color=color, label=MODEL_LABELS[model])
        ax.text(-0.03, y[0], f"{base.method} {y[0]:.2f}", ha="right", va="center", fontsize=8)
        ax.text(1.03, y[1], f"{newg.method} {y[1]:.2f} ({y[1]-y[0]:+.2f})", ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.35, 1.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["Best baseline", "Best NewG"])
    ax.set_ylabel("F1 (%)")
    ax.set_title("MuSiQue: F1 Lift From Strongest Baseline to Best NewG", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    return save(fig, "positive_musique_best_f1_slope.png")


def draw_musique_top_methods_ranked(rows: list[MainRow]) -> Path:
    retriever_pairs = [
        ("HippoRAG", "NewG hippo+bm25", "NewG hippo+vdb"),
        ("ToG", "NewG tog+bm25", "NewG tog+vdb"),
        ("RAPTOR", "NewG raptor+bm25", "NewG raptor+vdb"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.4), sharex=True)
    for ax, model in zip(axes, MODELS):
        by_method = method_map(rows, "MuSiQue", model)
        labels = []
        vals = []
        annotations = []
        colors = []
        for base_name, bm25_name, vdb_name in retriever_pairs:
            base = by_method[base_name]
            bm25 = by_method[bm25_name]
            vdb = by_method[vdb_name]
            labels.extend([base_name, bm25_name.replace("NewG ", "N "), vdb_name.replace("NewG ", "N ")])
            vals.extend([base.f1, bm25.f1, vdb.f1])
            annotations.extend(
                [
                    f"{base.f1:.2f}",
                    f"{bm25.f1:.2f} ({bm25.f1 - base.f1:+.2f})",
                    f"{vdb.f1:.2f} ({vdb.f1 - base.f1:+.2f})",
                ]
            )
            colors.extend(["#6B7280", "#2A9D8F", "#B35C1E"])
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color=colors, edgecolor="#333333", linewidth=0.4)
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(MODEL_LABELS[model])
        for bar, value, text in zip(bars, vals, annotations):
            ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2, text, va="center", fontsize=7.2)
        for boundary in [2.5, 5.5]:
            ax.axhline(boundary, color="#DDE4EC", linewidth=0.8)
        style_axes(ax, grid_axis="x")
    axes[0].set_xlabel("F1 (%)")
    axes[1].set_xlabel("F1 (%)")
    axes[2].set_xlabel("F1 (%)")
    legend_handles = [
        Line2D([0], [0], color="#6B7280", lw=8, label="Original baseline"),
        Line2D([0], [0], color="#2A9D8F", lw=8, label="Paired NewG + BM25"),
        Line2D([0], [0], color="#B35C1E", lw=8, label="Paired NewG + VDB"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("MuSiQue: Paired Baselines and NewG Combinations by Model", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    return save(fig, "positive_musique_top_methods_ranked.png")


def draw_musique_radar(rows: list[MainRow]) -> Path:
    categories = ["Accuracy", "EM", "Precision", "Recall", "F1"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, 3, subplot_kw={"polar": True}, figsize=(12.6, 4.2))
    for ax, model in zip(axes, MODELS):
        base = best_by_arch(rows, "MuSiQue", model, "baseline")
        newg = best_by_arch(rows, "MuSiQue", model, "newg")
        for row, label, color, alpha in [
            (base, f"Best baseline\n{base.method}", "#6B7280", 0.12),
            (newg, f"Best NewG\n{newg.method}", "#2A9D8F", 0.22),
        ]:
            vals = [row.accuracy, row.em, row.precision, row.recall, row.f1]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, linewidth=2, label=label)
            ax.fill(angles, vals, color=color, alpha=alpha)
        ax.set_title(MODEL_LABELS[model], y=1.10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 40)
        ax.set_yticks([10, 20, 30, 40])
        ax.set_yticklabels(["10", "20", "30", "40"], fontsize=7)
        ax.grid(color="#D7DDE3", linewidth=0.7)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.25, -0.25), frameon=False, fontsize=8)
    fig.suptitle("MuSiQue Metric Profile: Best NewG vs Strongest Baseline", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "positive_musique_metric_radar.png")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def generate_report(rows: list[MainRow], loo_tables: dict[str, list[dict[str, float | str]]], outputs: list[Path]) -> None:
    main_rows: list[list[str]] = []
    for dataset in DATASETS:
        for model in MODELS:
            base = best_by_arch(rows, dataset, model, "baseline")
            newg = best_by_arch(rows, dataset, model, "newg")
            main_rows.append(
                [
                    dataset,
                    MODEL_LABELS[model],
                    f"{base.method} ({base.f1:.2f})",
                    f"{newg.method} ({newg.f1:.2f})",
                    f"{newg.accuracy - base.accuracy:+.2f}",
                    f"{newg.em - base.em:+.2f}",
                    f"{newg.f1 - base.f1:+.2f}",
                ]
            )

    counterpart_rows: list[list[str]] = []
    for dataset in DATASETS:
        for model in MODELS:
            by_method = method_map(rows, dataset, model)
            deltas = [by_method[newg].f1 - by_method[base].f1 for newg, base, _ in COUNTERPARTS]
            counterpart_rows.append(
                [
                    dataset,
                    MODEL_LABELS[model],
                    f"{sum(d > 0 for d in deltas)}/6",
                    f"{sum(deltas) / len(deltas):+.2f}",
                    f"{min(deltas):+.2f}",
                    f"{max(deltas):+.2f}",
                ]
            )

    old_rows: list[list[str]] = []
    for i, method in enumerate(OLD_ABLATION_METHODS):
        if i == 0:
            old_rows.append([method, "66.00", "24.00", "27.15", "reference"])
            continue
        old_rows.append(
            [
                method,
                f"{OLD_ABLATION['accuracy'][i]:.2f} ({OLD_ABLATION['accuracy'][i] - OLD_ABLATION['accuracy'][0]:+.2f})",
                f"{OLD_ABLATION['em'][i]:.2f} ({OLD_ABLATION['em'][i] - OLD_ABLATION['em'][0]:+.2f})",
                f"{OLD_ABLATION['f1'][i]:.2f} ({OLD_ABLATION['f1'][i] - OLD_ABLATION['f1'][0]:+.2f})",
                "mixed",
            ]
        )

    loo_rows: list[list[str]] = []
    for dataset in DATASETS:
        full = next(r for r in loo_tables[dataset] if r["variant"] == "full")
        for row in loo_tables[dataset]:
            variant = str(row["variant"])
            if variant == "full":
                continue
            loo_rows.append(
                [
                    dataset,
                    LOO_COMPONENT_LABELS.get(variant, variant),
                    f"{float(full['accuracy']) - float(row['accuracy']):+.2f}",
                    f"{float(full['em']) - float(row['em']):+.2f}",
                    f"{float(full['f1']) - float(row['f1']):+.2f}",
                ]
            )

    figure_rows = [[path.name, recommended_placement(path.name)] for path in outputs]

    text = f"""# Experiment Result Organization

This report keeps the full result structure, while separating results that are strong enough for main-paper narrative from mixed or negative findings better suited for appendix/limitations.

## Main Experiments: Best Baseline vs Best NewG

{markdown_table(["Dataset", "Model", "Best baseline F1", "Best NewG F1", "Delta Acc", "Delta EM", "Delta F1"], main_rows)}

Interpretation:

- Strong positive: MuSiQue across all three models.
- Mild positive: PopQA with DeepSeek and GPT-4o-mini by F1, but accuracy/EM are flat or slightly lower.
- Mixed/negative: PopQA with Gemini when compared against the strongest single baseline.

## NewG Variants vs Paired Retriever Baselines

Each NewG variant is compared with the baseline that supplies its graph retriever family: HippoRAG, ToG, or RAPTOR.

{markdown_table(["Dataset", "Model", "Wins", "Avg Delta F1", "Worst Delta", "Best Delta"], counterpart_rows)}

Interpretation:

- This is a good positive summary because most dataset/model combinations have 4 or 5 wins out of 6.
- It is stronger than claiming every individual NewG variant wins.

## PopQA Incremental Ablation

This is the older PopQA ablation behind `popqa_ablation_*`.

{markdown_table(["Variant", "Accuracy", "EM", "F1", "Narrative"], old_rows)}

Interpretation:

- It is not a clean monotonic module-addition story.
- It does show that answer-control variants greatly increase EM/precision, but accuracy drops.
- Prefer appendix unless the paper explicitly discusses the accuracy-EM tradeoff.

## Leave-One-Out Ablation: Component Contribution

Values are `Full NewG - ablated variant`; positive means removing the component hurts.

{markdown_table(["Dataset", "Component", "Acc contribution", "EM contribution", "F1 contribution"], loo_rows)}

Main-paper-safe messages:

- Critic is consistently useful, especially on MuSiQue.
- Answer normalizer is important for F1/EM stability, even when raw accuracy can move in the opposite direction.
- Router, re-generator, and disambiguation are not consistently positive in this Gemini hippo+bm25 leave-one-out setting.

## Recommended Positive Figure Set

{markdown_table(["Figure", "Suggested placement"], figure_rows)}

## Paper Narrative Recommendation

Use MuSiQue as the main positive evidence: NewG shows clear gains on complex multi-hop QA across all three models. Use PopQA as a robustness or near-parity result on simpler factoid QA, not as the main performance claim. For ablations, focus the main text on critic and answer normalizer; put full mixed ablation charts in appendix or present them as limitations.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def recommended_placement(name: str) -> str:
    if name in {
        "positive_musique_best_newg_metric_gains.png",
        "positive_musique_best_f1_slope.png",
        "positive_loo_critic_normalizer_metric_drop.png",
        "positive_loo_key_component_f1_drop.png",
        "positive_musique_method_f1_heatmap.png",
        "positive_musique_top_methods_ranked.png",
    }:
        return "main text"
    if name in {
        "positive_best_newg_vs_best_baseline_f1.png",
        "positive_best_newg_gain_matrix.png",
        "positive_counterpart_win_count.png",
        "positive_counterpart_f1_delta_heatmap.png",
    }:
        return "main text or appendix"
    return "appendix / visual summary"


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    rows = read_main_rows()
    loo_tables = read_loo_tables()
    outputs = [
        draw_best_newg_vs_baseline_f1(rows),
        draw_musique_metric_gains(rows),
        draw_best_gain_matrix(rows),
        draw_counterpart_delta_heatmap(rows),
        draw_counterpart_win_rate(rows),
        draw_musique_method_heatmap(rows),
        draw_loo_component_contribution(loo_tables),
        draw_loo_key_component_metric_drop(loo_tables),
        draw_loo_key_component_f1_drop(loo_tables),
        draw_musique_slope(rows),
        draw_musique_top_methods_ranked(rows),
        draw_musique_radar(rows),
    ]
    generate_report(rows, loo_tables, outputs)
    for output in outputs:
        print(output)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
