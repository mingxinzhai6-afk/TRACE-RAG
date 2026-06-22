from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collect_experiment_results import METHOD_SPECS, MODELS, collect_result_map


OUT_DIR = Path("figures") / "main_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [method_label for _, method_label, _ in METHOD_SPECS]
METRICS = {"Accuracy": "accuracy", "EM": "em", "F1": "f1"}


def load_data() -> dict[str, dict[str, list[float | None]]]:
    rows = collect_result_map(
        "MuSiQue",
        include_static=True,
        prefer_static=True,
        prefer_static_on_bad_n=True,
    )
    data: dict[str, dict[str, list[float | None]]] = {}
    for model in MODELS:
        data[model] = {}
        for label, metric_key in METRICS.items():
            values = []
            for _, method_label, _ in METHOD_SPECS:
                row = rows.get((model, method_label))
                values.append(None if row is None else row.metrics.get(metric_key))
            data[model][label] = values
    return data


def grouped_bars(ax, data: dict[str, dict[str, list[float | None]]], metric: str) -> None:
    x = np.arange(len(METHODS))
    width = 0.25
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for idx, model in enumerate(MODELS):
        offset = (idx - 1) * width
        values = data[model][metric]
        plot_values = [np.nan if value is None else value for value in values]
        bars = ax.bar(
            x + offset,
            plot_values,
            width,
            label=model,
            color=colors[idx],
            edgecolor="#333333",
            linewidth=0.4,
        )
        for bar, value in zip(bars, values):
            if value is None:
                continue
            if metric == "F1" or value >= 20:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.45,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 42 if metric != "F1" else 36)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=35, ha="right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    data = load_data()

    fig, axes = plt.subplots(3, 1, figsize=(15, 13.5))
    for ax, metric in zip(axes, ["Accuracy", "EM", "F1"]):
        grouped_bars(ax, data, metric)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.suptitle("MuSiQue Cross-Model Comparison by Method", fontsize=15, fontweight="bold", y=0.992)
    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.06, right=0.99, hspace=0.72)
    fig.savefig(OUT_DIR / "musique_cross_model_by_method.png", dpi=220)

    fig2, ax2 = plt.subplots(figsize=(15, 5.3))
    grouped_bars(ax2, data, "F1")
    ax2.legend(loc="upper right", frameon=False)
    ax2.set_title("MuSiQue F1 Cross-Model Comparison by Method", fontsize=14, fontweight="bold")
    fig2.subplots_adjust(top=0.88, bottom=0.28, left=0.06, right=0.99)
    fig2.savefig(OUT_DIR / "musique_cross_model_f1.png", dpi=220)

    print(OUT_DIR / "musique_cross_model_by_method.png")
    print(OUT_DIR / "musique_cross_model_f1.png")


if __name__ == "__main__":
    main()
