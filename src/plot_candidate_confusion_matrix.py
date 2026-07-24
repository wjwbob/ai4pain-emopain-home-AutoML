from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SWEEP_DIR = Path("baseline_outputs_fixed_model_sweep")
DEFAULT_CANDIDATE = "random_forest"
DEFAULT_ELITE_SUMMARY = Path("baseline_outputs_csv_elit") / "nested_logo_summary_metrics_from_csv.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a row-normalized confusion matrix.")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=DEFAULT_SWEEP_DIR,
        help="Directory containing per-candidate fixed-model sweep outputs.",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=DEFAULT_CANDIDATE,
        help="Candidate folder name, e.g. random_forest or lightgbm.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to the candidate folder.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional direct summary JSON path. Overrides --sweep-dir and --candidate.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional title label for summary files without candidate_name.",
    )
    return parser.parse_args()


def load_summary_metrics(sweep_dir: Path, candidate: str) -> dict:
    summary_path = sweep_dir / candidate / "nested_logo_summary_metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary metrics file not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_summary_metrics_from_path(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary metrics file not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def row_normalize(confusion_matrix: np.ndarray) -> np.ndarray:
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    return np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums != 0,
    )


def annotate_heatmap(ax: plt.Axes, data: np.ndarray) -> None:
    threshold = float(np.nanmax(data)) * 0.55 if data.size else 0.0
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            color = "white" if value > threshold else "#10243e"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=42,
            )


def draw_normalized_confusion_matrix(
    summary_metrics: dict,
    output_path: Path,
    label: str | None = None,
) -> None:
    candidate_name = label or str(summary_metrics.get("candidate_name", "model"))
    weighted_f1 = float(summary_metrics["weighted_f1"])
    class_order = [str(label) for label in summary_metrics["class_order"]]
    confusion_matrix = np.asarray(summary_metrics["confusion_matrix"], dtype=float)
    normalized_confusion_matrix = row_normalize(confusion_matrix)

    fig, ax = plt.subplots(figsize=(16, 12), dpi=220)
    image = ax.imshow(
        normalized_confusion_matrix,
        cmap=plt.cm.Blues,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )
    annotate_heatmap(ax, normalized_confusion_matrix)

    ax.set_title(
        f"Row-normalized Confusion Matrix",
        fontsize=56,
        pad=16,
    )
    ax.set_xticks(range(len(class_order)))
    ax.set_yticks(range(len(class_order)))
    ax.set_xticklabels(class_order, fontsize=38)
    ax.set_yticklabels(class_order, fontsize=38)
    ax.set_xlabel("Predicted label", fontsize=46)
    ax.set_ylabel("True label", fontsize=46)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.summary_path is not None:
        summary_metrics = load_summary_metrics_from_path(args.summary_path)
        default_output = args.summary_path.parent / "normalized_confusion_matrix.png"
    else:
        summary_metrics = load_summary_metrics(args.sweep_dir, args.candidate)
        default_output = args.sweep_dir / args.candidate / f"{args.candidate}_confusion_matrices.png"
    output_path = args.output if args.output is not None else default_output
    draw_normalized_confusion_matrix(summary_metrics, output_path, label=args.label)
    print(f"Saved row-normalized confusion matrix to: {output_path}")


if __name__ == "__main__":
    main()
