from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collect_experiment_results import METHOD_SPECS, MODELS, collect_result_map


OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

BASELINES = [method_label for _, method_label, architecture in METHOD_SPECS if architecture == "baseline"]
NEWG = [method_label for _, method_label, architecture in METHOD_SPECS if architecture == "newg"]
METHODS = [method_label for _, method_label, _ in METHOD_SPECS]
METRICS = ["Accuracy", "EM", "F1"]
METRIC_KEYS = {"Accuracy": "accuracy", "EM": "em", "F1": "f1"}

MODEL_SLUG = {
    "deepseek-v3.2": "deepseek",
    "gpt-4o-mini": "gpt4omini",
    "gemini-2.5-flash-lite": "gemini",
}


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
        for label, metric_key in METRIC_KEYS.items():
            values = []
            for _, method_label, _ in METHOD_SPECS:
                row = rows.get((model, method_label))
                values.append(None if row is None else row.metrics.get(metric_key))
            data[model][label] = values
    return data


def draw_model_chart(model: str, data: dict[str, dict[str, list[float | None]]]) -> Path:
    values = data[model]
    x = np.arange(len(METHODS))
    width = 0.24
    colors = {
        "Accuracy": "#4C78A8",
        "EM": "#F58518",
        "F1": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.axvspan(-0.5, len(BASELINES) - 0.5, color="#F2F4F7", zorder=0)
    ax.axvspan(len(BASELINES) - 0.5, len(METHODS) - 0.5, color="#EEF7EE", zorder=0)
    ax.axvline(len(BASELINES) - 0.5, color="#444444", linewidth=1.2)

    for idx, metric in enumerate(METRICS):
        offset = (idx - 1) * width
        metric_values = values[metric]
        plot_values = [np.nan if value is None else value for value in metric_values]
        bars = ax.bar(
            x + offset,
            plot_values,
            width,
            label=metric,
            color=colors[metric],
            edgecolor="#333333",
            linewidth=0.35,
        )
        for bar, value in zip(bars, metric_values):
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

    ax.text(
        (len(BASELINES) - 1) / 2,
        39,
        "Baselines",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        len(BASELINES) + (len(NEWG) - 1) / 2,
        39,
        "NewG variants",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    ax.set_title(f"MuSiQue Baselines vs NewG ({model})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 42)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=32, ha="right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.87, bottom=0.27)
    out = OUT_DIR / f"musique_{MODEL_SLUG[model]}_baseline_vs_newg.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def clean_values(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def mean_or_nan(values: list[float | None]) -> float:
    clean = clean_values(values)
    return sum(clean) / len(clean) if clean else float("nan")


def max_or_nan(values: list[float | None]) -> float:
    clean = clean_values(values)
    return max(clean) if clean else float("nan")


def draw_f1_summary(data: dict[str, dict[str, list[float | None]]]) -> Path:
    models = list(data.keys())
    baseline_avg = []
    newg_avg = []
    best_baseline = []
    best_newg = []

    for model in models:
        f1 = data[model]["F1"]
        baseline_vals = f1[: len(BASELINES)]
        newg_vals = f1[len(BASELINES):]
        baseline_avg.append(mean_or_nan(baseline_vals))
        newg_avg.append(mean_or_nan(newg_vals))
        best_baseline.append(max_or_nan(baseline_vals))
        best_newg.append(max_or_nan(newg_vals))

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ("Baseline avg F1", baseline_avg, "#9E9E9E", -1.5),
        ("NewG avg F1", newg_avg, "#54A24B", -0.5),
        ("Best baseline F1", best_baseline, "#4C78A8", 0.5),
        ("Best NewG F1", best_newg, "#F58518", 1.5),
    ]
    for label, vals, color, mult in series:
        bars = ax.bar(x + mult * width, vals, width, label=label, color=color, edgecolor="#333333", linewidth=0.35)
        for bar, value in zip(bars, vals):
            if np.isnan(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.45, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("MuSiQue F1 Summary: Baselines vs NewG", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 36)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.26)

    out = OUT_DIR / "musique_f1_baseline_vs_newg_summary.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    data = load_data()
    outputs = [draw_model_chart(model, data) for model in data]
    outputs.append(draw_f1_summary(data))
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
