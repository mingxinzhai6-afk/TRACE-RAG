from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("figures") / "ablation_experiments" / "incremental_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "Simple",
    "+Router",
    "+Re-Generator",
    "+Critic & Commendor",
    "w/o Commendor",
    "Single judge/voter",
    "Full NewG",
]

DATA = {
    "Accuracy": [66.00, 66.00, 66.00, 66.50, 58.50, 60.00, 59.50],
    "EM": [24.00, 24.00, 19.50, 20.00, 57.50, 58.50, 58.50],
    "Precision": [42.16, 42.16, 37.78, 38.38, 58.83, 60.08, 59.83],
    "Recall": [26.49, 26.49, 25.78, 25.88, 21.57, 21.94, 21.69],
    "F1": [27.15, 27.15, 25.92, 26.05, 27.28, 27.81, 27.50],
}

COLORS = {
    "Accuracy": "#4C78A8",
    "EM": "#F58518",
    "Precision": "#54A24B",
    "Recall": "#B279A2",
    "F1": "#E45756",
}


def style_axes(ax) -> None:
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_metric_profile() -> Path:
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    ax.axvspan(-0.5, 3.5, color="#F6F7F9", zorder=0)
    ax.axvspan(3.5, len(METHODS) - 0.5, color="#F1F8F3", zorder=0)
    ax.axvline(3.5, color="#666666", linewidth=1.0)

    for metric in ["Accuracy", "EM", "Precision", "Recall", "F1"]:
        vals = DATA[metric]
        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=COLORS[metric],
            label=metric,
        )
        for i, value in enumerate(vals):
            if metric in {"EM", "F1"} or i in {0, len(vals) - 1}:
                ax.text(i, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("PopQA Ablation Metric Profile", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 72)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=28, ha="right")
    ax.text(1.5, 69, "incremental modules", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(5.0, 69, "answer-control variants", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5, frameon=False)
    style_axes(ax)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.31)
    out = OUT_DIR / "popqa_ablation_metric_profile.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def draw_delta_vs_simple() -> Path:
    metrics = ["Accuracy", "EM", "Precision", "Recall", "F1"]
    methods = METHODS[1:]
    y = np.arange(len(methods))
    height = 0.14

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.axvline(0, color="#333333", linewidth=1.0)

    for idx, metric in enumerate(metrics):
        baseline = DATA[metric][0]
        deltas = [value - baseline for value in DATA[metric][1:]]
        offset = (idx - 2) * height
        ax.barh(
            y + offset,
            deltas,
            height,
            label=metric,
            color=COLORS[metric],
            edgecolor="#333333",
            linewidth=0.35,
        )

    ax.set_title("PopQA Ablation Delta vs Simple", fontsize=15, fontweight="bold")
    ax.set_xlabel("Score change from Simple (percentage points)")
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(-12, 40)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.12)
    out = OUT_DIR / "popqa_ablation_delta_vs_simple.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def draw_accuracy_em_tradeoff() -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    accuracy = DATA["Accuracy"]
    em = DATA["EM"]
    f1 = DATA["F1"]

    sizes = [260 + 25 * (value - min(f1)) for value in f1]
    scatter = ax.scatter(
        accuracy,
        em,
        s=sizes,
        c=f1,
        cmap="RdYlGn",
        vmin=25.5,
        vmax=28.0,
        edgecolor="#333333",
        linewidth=0.7,
        zorder=3,
    )

    # Keep marker coordinates exact, but use point-offset annotations so close
    # methods remain readable. Simple and +Router have identical coordinates.
    annotations = [
        ("Simple / +Router", 66.00, 24.00, (-10, 28), "right"),
        ("+Re-Generator", 66.00, 19.50, (-18, -20), "right"),
        ("+Critic & Commendor", 66.50, 20.00, (-22, 16), "right"),
        ("w/o Commendor", 58.50, 57.50, (16, -2), "left"),
        ("Single judge/voter", 60.00, 58.50, (16, 8), "left"),
        ("Full NewG", 59.50, 58.50, (16, -14), "left"),
    ]
    for label, x, y, offset, ha in annotations:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-",
                "color": "#555555",
                "linewidth": 0.6,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    ax.set_title("PopQA Accuracy-EM Tradeoff", fontsize=15, fontweight="bold")
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("EM (%)")
    ax.set_xlim(56, 68.5)
    ax.set_ylim(16, 62.5)
    style_axes(ax)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("F1 (%)")

    ax.text(
        0.01,
        0.02,
        "Markers show exact Accuracy/EM coordinates; color and size encode F1.",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        ha="left",
        va="bottom",
    )

    fig.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.12)
    out = OUT_DIR / "popqa_ablation_accuracy_em_tradeoff.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    outputs = [
        draw_metric_profile(),
        draw_delta_vs_simple(),
        draw_accuracy_em_tradeoff(),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
