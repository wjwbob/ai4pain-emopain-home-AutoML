from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from emopain_autogluon_baseline import (
    GROUP_COLUMN,
    LABEL_COLUMN,
    apply_smote_if_possible,
    rank_features_with_random_forest,
)


DEFAULT_METADATA_COLUMNS = [
    "source_file",
    "file_name",
    "cohort",
    "participant_id",
    "activity_type_id",
    "activity_instance_id",
    "group_id",
    "pain_score",
    "pain_class",
    "pain_class_original",
    "sampling_rate_hz",
    "segment_index",
    "num_frames",
    "nan_ratio",
    "window_duration_s",
    "step_duration_s",
    "window_size",
    "step_size",
    "num_windows",
    "torso_scale",
    "torso_scale_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot EmoPain@Home skeleton feature importance and permutation importance "
            "from the exported feature CSV."
        )
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "emopain_all_npy_features.csv",
        help="Feature CSV exported from export_emopain_features_to_csv.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "feature_importance_outputs",
        help="Directory where plots and summary tables will be written.",
    )
    parser.add_argument(
        "--pain-only",
        action="store_true",
        help="Restrict analysis to the pain cohort only.",
    )
    parser.add_argument(
        "--drop-constant-columns",
        action="store_true",
        help="Drop feature columns with a single unique numeric value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=256,
        help="Number of globally ranked features to keep for the plots and permutation analysis.",
    )
    parser.add_argument(
        "--plot-top-n",
        type=int,
        default=25,
        help="How many top features to show in the bar plots.",
    )
    parser.add_argument(
        "--permutation-max-features",
        type=int,
        default=30,
        help="Maximum number of globally ranked features to include in permutation importance.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=10,
        help="Number of shuffles per feature inside each held-out group.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap for LOGO folds, useful for smoke tests.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for feature ranking, SMOTE, and permutation shuffling.",
    )
    return parser.parse_args()


def prepare_feature_table(
    feature_csv: Path,
    pain_only: bool,
    drop_constant_columns: bool,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    feature_df = pd.read_csv(feature_csv)

    if "source_file" in feature_df.columns and "file_name" not in feature_df.columns:
        feature_df = feature_df.rename(columns={"source_file": "file_name"})

    if GROUP_COLUMN not in feature_df.columns:
        feature_df[GROUP_COLUMN] = (
            feature_df["participant_id"].astype(str)
            + "_"
            + feature_df["activity_type_id"].astype(str)
            + "_"
            + feature_df["activity_instance_id"].astype(str)
        )

    if pain_only and "cohort" in feature_df.columns:
        feature_df = feature_df[feature_df["cohort"] == "pain"].copy()

    feature_df = feature_df[feature_df[LABEL_COLUMN].notna()].copy()
    feature_df = feature_df.reset_index(drop=True)

    metadata_columns = [column for column in DEFAULT_METADATA_COLUMNS if column in feature_df.columns]
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [column for column in numeric_columns if column not in metadata_columns]

    if drop_constant_columns:
        feature_columns = [
            column
            for column in feature_columns
            if feature_df[column].nunique(dropna=False) > 1
        ]

    return feature_df, metadata_columns, feature_columns


def infer_feature_family(feature_name: str) -> str:
    if feature_name.startswith("torso_scale"):
        return "torso_scale"
    if feature_name.startswith("dist_") and "__body_center" in feature_name:
        return "distance_to_body_center"
    if feature_name.startswith("dist_"):
        return "pairwise_distance"
    if "_magnitude_jerk_" in feature_name:
        return "magnitude_jerk"
    if "_magnitude_acceleration_" in feature_name:
        return "magnitude_acceleration"
    if "_magnitude_velocity_" in feature_name:
        return "magnitude_velocity"
    if "_magnitude_raw_" in feature_name:
        return "magnitude_raw"
    if "_jerk_" in feature_name:
        return "axis_jerk"
    if "_acceleration_" in feature_name:
        return "axis_acceleration"
    if "_velocity_" in feature_name:
        return "axis_velocity"
    if "_raw_" in feature_name:
        return "axis_raw"
    return "other"


def make_feature_inventory(feature_columns: List[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature_name": feature_columns,
                "feature_family": [infer_feature_family(name) for name in feature_columns],
            }
        )
        .groupby("feature_family", as_index=False)
        .agg(n_columns=("feature_name", "size"))
        .sort_values(["n_columns", "feature_family"], ascending=[False, True])
        .reset_index(drop=True)
    )


def safe_balanced_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        return float(balanced_accuracy_score(y_true, y_pred))


def train_global_random_forest(
    feature_df: pd.DataFrame,
    selected_features: List[str],
    random_state: int,
) -> RandomForestClassifier:
    x_train = feature_df[selected_features].copy()
    y_train = feature_df[LABEL_COLUMN].copy()
    x_train, y_train = apply_smote_if_possible(x_train, y_train, random_state=random_state)

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def compute_rf_importance(
    model: RandomForestClassifier,
    selected_features: List[str],
) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "feature_name": selected_features,
            "importance": model.feature_importances_,
        }
    )
    importance_df["feature_family"] = importance_df["feature_name"].map(infer_feature_family)
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_logo_permutation_importance(
    feature_df: pd.DataFrame,
    selected_features: List[str],
    permutation_features: List[str],
    random_state: int,
    permutation_repeats: int,
    max_folds: int | None,
) -> pd.DataFrame:
    logo = LeaveOneGroupOut()
    groups = feature_df[GROUP_COLUMN].to_numpy()
    folds = list(logo.split(feature_df, groups=groups))
    if max_folds is not None:
        folds = folds[:max_folds]

    fold_rows: List[Dict[str, float | int | str]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df = feature_df.iloc[train_idx].reset_index(drop=True)
        test_df = feature_df.iloc[test_idx].reset_index(drop=True)

        x_train = train_df[selected_features].copy()
        y_train = train_df[LABEL_COLUMN].copy()
        x_test = test_df[selected_features].copy()
        y_test = test_df[LABEL_COLUMN].copy()

        x_train, y_train = apply_smote_if_possible(x_train, y_train, random_state=random_state)
        if y_train.nunique() < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        baseline_score = safe_balanced_accuracy(y_test, model.predict(x_test))
        rng = np.random.default_rng(seed=random_state + fold_idx)
        held_out_group = str(test_df.iloc[0][GROUP_COLUMN])

        for feature_name in permutation_features:
            repeat_scores = []
            for _ in range(permutation_repeats):
                x_permuted = x_test.copy()
                x_permuted[feature_name] = rng.permutation(x_permuted[feature_name].to_numpy())
                permuted_score = safe_balanced_accuracy(y_test, model.predict(x_permuted))
                repeat_scores.append(baseline_score - permuted_score)

            fold_rows.append(
                {
                    "outer_fold": fold_idx,
                    "held_out_group": held_out_group,
                    "n_test_segments": int(len(test_df)),
                    "feature_name": feature_name,
                    "feature_family": infer_feature_family(feature_name),
                    "baseline_balanced_accuracy": float(baseline_score),
                    "importance_mean": float(np.mean(repeat_scores)),
                    "importance_std": float(np.std(repeat_scores)),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return fold_df

    grouped_rows = []
    for (feature_name, feature_family), group in fold_df.groupby(
        ["feature_name", "feature_family"],
        sort=False,
    ):
        grouped_rows.append(
            {
                "feature_name": feature_name,
                "feature_family": feature_family,
                "importance_mean": float(group["importance_mean"].mean()),
                "importance_std": float(group["importance_mean"].std(ddof=0)),
                "weighted_importance_mean": float(
                    np.average(group["importance_mean"], weights=group["n_test_segments"])
                ),
                "n_folds": int(group["outer_fold"].nunique()),
            }
        )

    return (
        pd.DataFrame(grouped_rows)
        .sort_values("weighted_importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_top_features(
    importance_df: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    top_n: int,
) -> None:
    plot_df = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature_name"], plot_df[value_column], color="#4C78A8", edgecolor="black")
    plt.xlabel(value_column)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_family_summary(
    importance_df: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
) -> None:
    family_df = (
        importance_df.groupby("feature_family", as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
    )
    plt.figure(figsize=(9, 5))
    plt.bar(family_df["feature_family"], family_df[value_column], color="#F58518", edgecolor="black")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(value_column)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_df, metadata_columns, feature_columns = prepare_feature_table(
        feature_csv=args.feature_csv,
        pain_only=args.pain_only,
        drop_constant_columns=args.drop_constant_columns,
    )
    feature_inventory_df = make_feature_inventory(feature_columns)

    ranked_features = rank_features_with_random_forest(
        feature_df[feature_columns],
        feature_df[LABEL_COLUMN],
        random_state=args.random_state,
    )
    selected_top_k = int(min(max(1, args.top_k), len(ranked_features)))
    selected_features = ranked_features[:selected_top_k]
    permutation_feature_count = int(min(args.permutation_max_features, len(selected_features)))
    permutation_features = selected_features[:permutation_feature_count]

    rf_model = train_global_random_forest(
        feature_df=feature_df,
        selected_features=selected_features,
        random_state=args.random_state,
    )
    rf_importance_df = compute_rf_importance(rf_model, selected_features)
    permutation_df = compute_logo_permutation_importance(
        feature_df=feature_df,
        selected_features=selected_features,
        permutation_features=permutation_features,
        random_state=args.random_state,
        permutation_repeats=args.permutation_repeats,
        max_folds=args.max_folds,
    )

    feature_inventory_df.to_csv(args.output_dir / "feature_inventory_summary.csv", index=False)
    pd.DataFrame({"feature_name": ranked_features}).to_csv(
        args.output_dir / "ranked_feature_list.csv",
        index=False,
    )
    rf_importance_df.to_csv(args.output_dir / "rf_feature_importance.csv", index=False)
    if not permutation_df.empty:
        permutation_df.to_csv(args.output_dir / "permutation_importance_logo.csv", index=False)

    plot_top_features(
        rf_importance_df,
        value_column="importance",
        output_path=args.output_dir / "rf_feature_importance_top.png",
        title=f"Random Forest Feature Importance (top {min(args.plot_top_n, len(rf_importance_df))})",
        top_n=args.plot_top_n,
    )
    plot_family_summary(
        rf_importance_df,
        value_column="importance",
        output_path=args.output_dir / "rf_feature_importance_by_family.png",
        title="Random Forest Importance by Feature Family",
    )

    if not permutation_df.empty:
        plot_top_features(
            permutation_df,
            value_column="weighted_importance_mean",
            output_path=args.output_dir / "permutation_importance_top.png",
            title=(
                "Permutation Importance (LOGO balanced accuracy drop, "
                f"top {min(args.plot_top_n, len(permutation_df))})"
            ),
            top_n=args.plot_top_n,
        )
        plot_family_summary(
            permutation_df,
            value_column="weighted_importance_mean",
            output_path=args.output_dir / "permutation_importance_by_family.png",
            title="Permutation Importance by Feature Family",
        )

    summary = {
        "feature_csv": str(args.feature_csv),
        "output_dir": str(args.output_dir),
        "pain_only": bool(args.pain_only),
        "drop_constant_columns": bool(args.drop_constant_columns),
        "n_samples": int(len(feature_df)),
        "n_groups": int(feature_df[GROUP_COLUMN].nunique()),
        "n_metadata_columns": int(len(metadata_columns)),
        "n_feature_columns": int(len(feature_columns)),
        "selected_top_k": int(selected_top_k),
        "permutation_feature_count": int(permutation_feature_count),
        "feature_family_counts": feature_inventory_df.set_index("feature_family")["n_columns"].to_dict(),
    }
    (args.output_dir / "feature_importance_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved feature importance outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
