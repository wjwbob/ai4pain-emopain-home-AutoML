from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_ORDER = ["LP", "MP", "HP"]

PEREIRA_LOAO = np.array(
    [
        [38, 33, 12],
        [46, 179, 70],
        [6, 52, 90],
    ],
    dtype=float,
)

OUR_BASELINE = np.array(
    [
        [34, 24, 25],
        [30, 199, 66],
        [17, 72, 59],
    ],
    dtype=float,
)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )


def annotate_counts_and_rates(ax: plt.Axes, counts: np.ndarray) -> None:
    rates = row_normalize(counts)
    threshold = float(np.nanmax(rates)) * 0.55
    for row_idx in range(counts.shape[0]):
        for col_idx in range(counts.shape[1]):
            rate = rates[row_idx, col_idx]
            color = "white" if rate > threshold else "#16324f"
            ax.text(
                col_idx,
                row_idx,
                f"{int(counts[row_idx, col_idx])}\n{rate:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=17,
                fontweight="semibold",
            )


def annotate_difference(ax: plt.Axes, diff: np.ndarray) -> None:
    for row_idx in range(diff.shape[0]):
        for col_idx in range(diff.shape[1]):
            value = int(diff[row_idx, col_idx])
            if value > 0:
                text = f"+{value}"
            else:
                text = str(value)
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="#111827",
                fontsize=19,
                fontweight="semibold",
            )


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=18, pad=14, fontweight="semibold")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, fontsize=13)
    ax.set_yticklabels(CLASS_ORDER, fontsize=13)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def main() -> None:
    output_path = (
        Path(__file__).resolve().parents[1]
        / "pereira_loao_outputs"
        / "pereira_vs_ours_confusion_matrices.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.2),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )

    pereira_rates = row_normalize(PEREIRA_LOAO)
    ours_rates = row_normalize(OUR_BASELINE)
    diff = PEREIRA_LOAO - OUR_BASELINE

    image0 = axes[0].imshow(pereira_rates, cmap="Blues", vmin=0.0, vmax=1.0)
    annotate_counts_and_rates(axes[0], PEREIRA_LOAO)
    style_axis(axes[0], "Pereira Method\nOfficial LOAO Re-evaluation")

    axes[1].imshow(ours_rates, cmap="Blues", vmin=0.0, vmax=1.0)
    annotate_counts_and_rates(axes[1], OUR_BASELINE)
    style_axis(axes[1], "Our Baseline\nOfficial LOAO")

    vmax = float(np.max(np.abs(diff)))
    image2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    annotate_difference(axes[2], diff)
    style_axis(axes[2], "Difference\nPereira - Ours")

    fig.suptitle(
        "Confusion Matrix Comparison",
        fontsize=22,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Cells in the first two panels show count and row-normalized recall percentage.",
        ha="center",
        fontsize=11,
        color="#4b5563",
    )

    colorbar0 = fig.colorbar(image0, ax=axes[:2], fraction=0.022, pad=0.02)
    colorbar0.ax.set_ylabel("Row-normalized rate", fontsize=11)
    colorbar0.ax.tick_params(labelsize=10)
    colorbar2 = fig.colorbar(image2, ax=axes[2], fraction=0.046, pad=0.03)
    colorbar2.ax.set_ylabel("Count difference", fontsize=11)
    colorbar2.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
