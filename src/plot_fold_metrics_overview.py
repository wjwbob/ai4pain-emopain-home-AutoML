from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "baseline_outputs_csv_elit"
    / "nested_logo_fold_metrics_from_csv.csv"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "baseline_outputs_csv_elit"
    / "fold_level_weighted_f1_overview.png"
)
DEFAULT_SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "baseline_outputs_csv_elit"
    / "nested_logo_summary_metrics_from_csv.json"
)
SUPPORTED_METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "weighted_f1": "Weighted F1",
    "macro_f1": "Macro F1",
}
PRIMARY_BAR_COLOR = "#202020"
SECONDARY_BAR_COLOR = "#f0f0f0"
EDGE_COLOR = "#000000"
COUNT_BAR_EDGE_WIDTH = 1.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fold-level performance from nested_logo_fold_metrics_from_csv.csv "
            "using stacked bar charts for the selected metric and the number of test segments."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Input fold-metrics CSV path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination image path.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(SUPPORTED_METRICS),
        default="weighted_f1",
        help="Metric column to plot in the top panel.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Optional summary JSON used for the overall metric reference line.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    return parser.parse_args()


def build_default_title(metric_label: str, n_folds: int) -> str:
    return f"Fold-level baseline performance across {n_folds} outer folds"


def load_overall_metric(summary_path: Path, metric: str) -> float | None:
    if not summary_path.exists():
        return None

    summary_metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    if metric not in summary_metrics:
        return None
    return float(summary_metrics[metric])


def plot_fold_metrics(
    csv_path: Path,
    output_path: Path,
    metric: str,
    title: str | None,
    summary_path: Path,
) -> Path:
    fold_df = pd.read_csv(csv_path)
    required_columns = {"outer_fold", "n_test_segments", metric}
    missing = required_columns.difference(fold_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")

    fold_df = fold_df.sort_values("outer_fold").reset_index(drop=True)
    x_values = fold_df["outer_fold"].astype(int).to_numpy()
    metric_values = fold_df[metric].astype(float).to_numpy()
    test_segment_values = fold_df["n_test_segments"].astype(float).to_numpy()
    metric_label = SUPPORTED_METRICS[metric]
    overall_metric = load_overall_metric(summary_path, metric)
    reference_metric = overall_metric if overall_metric is not None else float(fold_df[metric].mean())
    reference_label = "overall" if overall_metric is not None else "mean"

    # Font size settings
    TITLE_SIZE = 42
    AXIS_LABEL_SIZE = 30
    TICK_LABEL_SIZE = 30
    LEGEND_SIZE = 28

    fig, (ax_metric, ax_count) = plt.subplots(
        2,
        1,
        figsize=(20, 12),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.15], "hspace": 0.30},
    )
    fig.subplots_adjust(top=0.82, hspace=0.30, left=0.10)

    metric_bar_container = ax_metric.bar(
        x_values,
        metric_values,
        width=0.62,
        color=PRIMARY_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        label=metric_label,
        zorder=3,
    )

    mean_handle = ax_metric.axhline(
        reference_metric,
        color=EDGE_COLOR,
        linestyle="--",
        linewidth=1.4,
        label=f"{metric_label} {reference_label} = {reference_metric:.3f}",
        zorder=4,
    )

    count_bar_container = ax_count.bar(
        x_values,
        test_segment_values,
        width=0.62,
        color=SECONDARY_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=COUNT_BAR_EDGE_WIDTH,
        label="No. test segments",
        zorder=3,
    )

    fig.suptitle(
        title or build_default_title(metric_label=metric_label, n_folds=len(fold_df)),
        fontsize=TITLE_SIZE,
        y=0.97,
    )

    ax_metric.set_ylabel(metric_label, fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax_count.set_ylabel("No. test segments", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax_count.set_xlabel("Outer fold", fontsize=AXIS_LABEL_SIZE, labelpad=12)

    ax_metric.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax_count.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax_count.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

    ax_metric.set_xlim(0.5, x_values.max() + 0.5)
    ax_metric.set_ylim(0.0, 1.04)
    ax_count.set_ylim(0, max(test_segment_values.max() * 1.10, 1))

    max_fold = int(x_values.max())
    tick_positions = [1] + list(range(5, max_fold + 1, 5))
    if max_fold not in tick_positions:
        tick_positions.append(max_fold)
    ax_count.set_xticks(tick_positions)

    ax_metric.grid(axis="y", linestyle="--", color="#8c8c8c", alpha=0.55, linewidth=0.7)
    ax_count.grid(axis="y", linestyle="--", color="#a6a6a6", alpha=0.45, linewidth=0.7)
    ax_metric.set_axisbelow(True)
    ax_count.set_axisbelow(True)

    legend_handles = [metric_bar_container, mean_handle, count_bar_container]
    legend_labels = [handle.get_label() for handle in legend_handles]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=True,
        fontsize=LEGEND_SIZE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = plot_fold_metrics(
        csv_path=args.csv_path.resolve(),
        output_path=args.output_path.resolve(),
        metric=args.metric,
        title=args.title,
        summary_path=args.summary_path.resolve(),
    )
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
