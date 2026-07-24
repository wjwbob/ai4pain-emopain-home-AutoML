from __future__ import annotations

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from emopain_data_utils import (
    collect_skeleton_files,
    describe_supported_layouts,
    load_skeleton_array,
    parse_emopain_filename,
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut


JOINT_NAMES: List[str] = [
    "hip",
    "mid_spine",
    "right_ankle",
    "right_elbow",
    "right_knee",
    "right_wrist",
]
PAIR_NAMES: List[Tuple[int, int]] = list(combinations(range(len(JOINT_NAMES)), 2))
ANGLE_TRIPLETS: Dict[str, Tuple[int, int, int]] = {
    "elbow_angle": (1, 3, 5),
    "knee_angle": (0, 4, 2),
    "torso_wrist_angle": (0, 1, 5),
    "torso_ankle_angle": (1, 0, 2),
}
METADATA_COLUMNS: List[str] = [
    "file_name",
    "participant_id",
    "activity_type_id",
    "activity_instance_id",
    "group_id",
    "pain_score",
    "pain_class",
    "sampling_rate_hz",
    "segment_index",
    "num_frames",
    "nan_ratio",
    "torso_scale",
]
LABEL_COLUMN = "pain_class"
GROUP_COLUMN = "group_id"
DEFAULT_TOP_K_GRID = [16, 32, 64, 128, 256]
CLASS_ORDER = ["LP", "MP", "HP"]
FEATURE_MISSING_SENTINEL = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EmoPain@Home pain classification baseline using skeleton features, "
            "AutoGluon, and nested leave-one-activity-instance-out validation."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root directory that contains the EmoPain pain folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "baseline_outputs",
        help="Directory where features and evaluation outputs will be written.",
    )
    parser.add_argument(
        "--features-cache",
        type=Path,
        default=None,
        help="Optional CSV cache for extracted features. Defaults to <output-dir>/pain_features.csv.",
    )
    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
        help="Recompute features even when the cache file already exists.",
    )
    parser.add_argument(
        "--top-k-grid",
        type=int,
        nargs="*",
        default=DEFAULT_TOP_K_GRID,
        help="Candidate feature counts for the inner-loop selector.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=45,
        help="AutoGluon time limit in seconds for each outer-fold fit.",
    )
    parser.add_argument(
        "--ag-presets",
        default="medium_quality",
        help="AutoGluon fit presets. Use medium_quality for a practical baseline.",
    )
    parser.add_argument(
        "--ag-model-preset",
        choices=["tabular_fast", "tree_only", "default"],
        default="tabular_fast",
        help="Model family subset used by AutoGluon in each outer fold.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for feature ranking, SMOTE, and model training.",
    )
    parser.add_argument(
        "--max-outer-folds",
        type=int,
        default=None,
        help="Optional limit for smoke tests. By default all 54 activity instances are evaluated.",
    )
    parser.add_argument(
        "--inner-max-groups",
        type=int,
        default=None,
        help="Optional limit for smoke tests inside the inner leave-one-group-out loop.",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Only extract features and write the cache file.",
    )
    return parser.parse_args()


def summarize_signal(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_iqr": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_rms": float("nan"),
        }

    q10, q25, q75, q90 = np.percentile(values, [10, 25, 75, 90])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_p10": float(q10),
        f"{prefix}_p90": float(q90),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(values)))),
    }


def interpolate_missing_values(array_2d: np.ndarray) -> np.ndarray:
    frame_df = pd.DataFrame(array_2d)
    frame_df = frame_df.interpolate(method="linear", axis=0, limit_area="inside")
    frame_df = frame_df.ffill().bfill()
    return frame_df.to_numpy(dtype=np.float64)


def compute_reference_scale(coords: np.ndarray) -> np.ndarray:
    root = coords[:, 0:1, :]
    root_to_other_joints = np.linalg.norm(coords[:, 1:, :] - root, axis=2)

    scales = np.full(coords.shape[0], np.nan, dtype=np.float64)
    for frame_idx in range(coords.shape[0]):
        valid_distances = root_to_other_joints[frame_idx]
        valid_distances = valid_distances[
            np.isfinite(valid_distances) & (valid_distances >= 1e-8)
        ]
        if valid_distances.size:
            scales[frame_idx] = float(np.median(valid_distances))
    return scales


def safe_angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    vec1 = a - b
    vec2 = c - b
    denom = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    cosine = np.sum(vec1 * vec2, axis=1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def build_signal_stack(sequence: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    velocity = np.gradient(sequence, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)
    return {
        "pos": sequence,
        "vel": velocity,
        "acc": acceleration,
        "jerk": jerk,
    }


def extract_segment_features(path: Path) -> Dict[str, object]:
    parsed_metadata = parse_emopain_filename(path)
    if parsed_metadata["cohort"] != "pain":
        raise ValueError(f"Expected a pain-cohort file, received {path.name}")

    metadata = {
        "file_name": str(parsed_metadata["source_file"]),
        "participant_id": str(parsed_metadata["participant_id"]),
        "activity_type_id": int(parsed_metadata["activity_type_id"]),
        "activity_instance_id": int(parsed_metadata["activity_instance_id"]),
        "group_id": str(parsed_metadata["group_id"]),
        "pain_score": float(parsed_metadata["pain_score"]),
        "pain_class": str(parsed_metadata["pain_class"]),
        "sampling_rate_hz": float(parsed_metadata["sampling_rate_hz"]),
        "segment_index": int(parsed_metadata["segment_index"]),
    }

    raw = load_skeleton_array(path)
    if raw.ndim != 2 or raw.shape[1] != 18:
        raise ValueError(f"Expected shape (T, 18), received {raw.shape} for {path.name}")

    metadata["num_frames"] = int(raw.shape[0])
    metadata["nan_ratio"] = float(np.isnan(raw).sum() / raw.size)

    filled = interpolate_missing_values(raw)
    coords = filled.reshape(filled.shape[0], len(JOINT_NAMES), 3)

    root = coords[:, 0:1, :]
    torso_distances = np.linalg.norm(coords[:, 1, :] - coords[:, 0, :], axis=1)
    fallback_scale = compute_reference_scale(coords)
    torso_distances = np.where(
        np.isfinite(torso_distances) & (torso_distances > 1e-8),
        torso_distances,
        fallback_scale,
    )
    valid_torso = torso_distances[np.isfinite(torso_distances) & (torso_distances > 1e-8)]
    torso_scale = float(np.median(valid_torso)) if valid_torso.size else 1.0
    metadata["torso_scale"] = torso_scale

    normalized = (coords - root) / torso_scale
    dt = 1.0 / float(metadata["sampling_rate_hz"])
    signal_stack = build_signal_stack(normalized, dt=dt)

    features: Dict[str, object] = dict(metadata)

    for signal_name, signal_values in signal_stack.items():
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            joint_xyz = signal_values[:, joint_idx, :]
            joint_magnitude = np.linalg.norm(joint_xyz, axis=1)

            for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                prefix = f"{joint_name}_{signal_name}_{axis_name}"
                features.update(summarize_signal(joint_xyz[:, axis_idx], prefix))

            features.update(
                summarize_signal(joint_magnitude, f"{joint_name}_{signal_name}_magnitude")
            )

    for joint_a, joint_b in PAIR_NAMES:
        pair_name = f"{JOINT_NAMES[joint_a]}__{JOINT_NAMES[joint_b]}_distance"
        pair_distance = np.linalg.norm(
            normalized[:, joint_a, :] - normalized[:, joint_b, :], axis=1
        )
        features.update(summarize_signal(pair_distance, pair_name))

    for angle_name, (joint_a, joint_b, joint_c) in ANGLE_TRIPLETS.items():
        angle_values = safe_angle_degrees(
            normalized[:, joint_a, :],
            normalized[:, joint_b, :],
            normalized[:, joint_c, :],
        )
        features.update(summarize_signal(angle_values, angle_name))

    wrist_height = normalized[:, 5, 1] - normalized[:, 1, 1]
    ankle_height = normalized[:, 2, 1] - normalized[:, 0, 1]
    features.update(summarize_signal(wrist_height, "wrist_relative_height"))
    features.update(summarize_signal(ankle_height, "ankle_relative_height"))

    return features


def extract_feature_table(dataset_root: Path, cache_path: Path, force: bool) -> pd.DataFrame:
    if cache_path.exists() and not force:
        print(f"Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    feature_rows: List[Dict[str, object]] = []
    files = [
        path
        for path in collect_skeleton_files(dataset_root, include_healthy=False)
        if path.stem.startswith("P")
    ]
    if not files:
        raise FileNotFoundError(
            "No pain skeleton files were found. " + describe_supported_layouts()
        )

    print(f"Extracting features from {len(files)} pain segments...")

    for idx, data_path in enumerate(files, start=1):
        if idx % 50 == 0 or idx == len(files):
            print(f"  processed {idx}/{len(files)} segments")
        feature_rows.append(extract_segment_features(data_path))

    feature_df = pd.DataFrame(feature_rows)
    feature_df = feature_df.sort_values(
        ["participant_id", "activity_type_id", "activity_instance_id", "segment_index"]
    ).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(cache_path, index=False)
    print(f"Saved feature cache to {cache_path}")
    return feature_df


def get_feature_columns(feature_df: pd.DataFrame) -> List[str]:
    return [column for column in feature_df.columns if column not in METADATA_COLUMNS]


def fit_minmax_scaler(feature_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    numeric_df = feature_df.astype(np.float64)
    return numeric_df.min(axis=0, skipna=True), numeric_df.max(axis=0, skipna=True)


def minmax_scale_features(
    feature_df: pd.DataFrame,
    min_values: pd.Series,
    max_values: pd.Series,
) -> pd.DataFrame:
    scaled = feature_df.astype(np.float64).copy()

    for column in scaled.columns:
        column_values = scaled[column].to_numpy(dtype=np.float64, copy=True)
        column_min = float(min_values[column])
        column_max = float(max_values[column])

        if not np.isfinite(column_min) or not np.isfinite(column_max):
            column_values[:] = np.nan
        else:
            finite_mask = np.isfinite(column_values)
            if column_max - column_min < 1e-12:
                column_values[finite_mask] = 0.0
            else:
                column_values[finite_mask] = (
                    (column_values[finite_mask] - column_min) / (column_max - column_min)
                )
                column_values[finite_mask] = np.clip(column_values[finite_mask], 0.0, 1.0)
            column_values[~finite_mask] = np.nan

        scaled[column] = column_values

    return scaled


def prepare_feature_matrices(
    train_features: pd.DataFrame,
    *other_feature_frames: pd.DataFrame,
) -> Tuple[pd.DataFrame, ...]:
    min_values, max_values = fit_minmax_scaler(train_features)
    prepared_frames: List[pd.DataFrame] = []

    for feature_frame in (train_features, *other_feature_frames):
        scaled = minmax_scale_features(feature_frame, min_values, max_values)
        prepared_frames.append(scaled.fillna(FEATURE_MISSING_SENTINEL))

    return tuple(prepared_frames)


def apply_smote_if_possible(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    class_counts = y_train.value_counts()
    if y_train.nunique() < 2 or class_counts.min() < 2:
        return x_train, y_train

    k_neighbors = int(min(5, class_counts.min() - 1))
    if k_neighbors < 1:
        return x_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    return pd.DataFrame(x_resampled, columns=x_train.columns), pd.Series(y_resampled, name=y_train.name)


def rank_features_with_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> List[str]:
    if y_train.nunique() < 2:
        return list(x_train.columns)

    selector = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    selector.fit(x_train, y_train)
    ranked_indices = np.argsort(selector.feature_importances_)[::-1]
    return list(x_train.columns[ranked_indices])


def build_top_k_grid(feature_count: int, requested_grid: Sequence[int]) -> List[int]:
    cleaned = []
    for value in requested_grid:
        if value <= 0:
            continue
        cleaned.append(min(int(value), feature_count))

    if not cleaned:
        cleaned = [feature_count]

    # if feature_count not in cleaned:
    #     cleaned.append(feature_count)

    return sorted(set(cleaned))


def select_top_k_via_nested_groups(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    top_k_grid: Sequence[int],
    random_state: int,
    inner_max_groups: int | None,
) -> Tuple[int, pd.DataFrame]:
    group_splitter = LeaveOneGroupOut()
    groups = train_df[GROUP_COLUMN].to_numpy()
    folds = list(group_splitter.split(train_df, groups=groups))
    if inner_max_groups is not None:
        folds = folds[:inner_max_groups]

    candidate_top_k = build_top_k_grid(len(feature_columns), top_k_grid)
    summary_rows: List[Dict[str, object]] = []

    if len(folds) == 0:
        return candidate_top_k[0], pd.DataFrame(columns=["top_k", "mean_weighted_f1", "mean_macro_f1"])

    for top_k in candidate_top_k:
        weighted_scores: List[float] = []
        macro_scores: List[float] = []

        for inner_train_idx, inner_val_idx in folds:
            inner_train = train_df.iloc[inner_train_idx]
            inner_val = train_df.iloc[inner_val_idx]

            x_inner_train = inner_train[list(feature_columns)]
            y_inner_train = inner_train[LABEL_COLUMN]
            x_inner_val = inner_val[list(feature_columns)]
            y_inner_val = inner_val[LABEL_COLUMN]
            x_inner_train, x_inner_val = prepare_feature_matrices(
                x_inner_train, x_inner_val
            )

            ranked_features = rank_features_with_random_forest(
                x_inner_train, y_inner_train, random_state=random_state
            )
            selected_features = ranked_features[:top_k]

            x_inner_train_selected = x_inner_train[selected_features]
            x_inner_val_selected = x_inner_val[selected_features]
            x_inner_train_selected, y_inner_train = apply_smote_if_possible(
                x_inner_train_selected, y_inner_train, random_state=random_state
            )

            if y_inner_train.nunique() < 2:
                y_pred = np.repeat(y_inner_train.iloc[0], len(y_inner_val))
            else:
                proxy_model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                )
                proxy_model.fit(x_inner_train_selected, y_inner_train)
                y_pred = proxy_model.predict(x_inner_val_selected)

            weighted_scores.append(
                f1_score(y_inner_val, y_pred, average="weighted", zero_division=0)
            )
            macro_scores.append(
                f1_score(y_inner_val, y_pred, average="macro", zero_division=0)
            )

        summary_rows.append(
            {
                "top_k": top_k,
                "mean_weighted_f1": float(np.mean(weighted_scores)),
                "mean_macro_f1": float(np.mean(macro_scores)),
                "inner_folds": len(weighted_scores),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_weighted_f1", "mean_macro_f1", "top_k"],
        ascending=[False, False, True],
    )
    best_top_k = int(summary_df.iloc[0]["top_k"])
    return best_top_k, summary_df


def get_autogluon_hyperparameters(model_preset: str) -> Dict[str, object] | None:
    if model_preset == "default":
        return None
    if model_preset == "tree_only":
        return {"RF": {}, "XT": {}}
    return {"GBM": {}, "CAT": {}, "XGB": {}, "RF": {}, "XT": {}}


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def json_dumps_compact(value: Any) -> str:
    return json.dumps(make_json_safe(value), ensure_ascii=True, sort_keys=True)


def prepare_leaderboard_for_export(
    leaderboard_df: pd.DataFrame,
    *,
    outer_fold: int,
    held_out_group: str,
    validation_group: str,
    best_model_name: str,
) -> pd.DataFrame:
    if leaderboard_df.empty:
        return pd.DataFrame(
            columns=[
                "outer_fold",
                "held_out_group",
                "validation_group",
                "best_model_name",
            ]
        )

    export_df = leaderboard_df.copy()
    for column in export_df.columns:
        export_df[column] = export_df[column].map(
            lambda value: json_dumps_compact(value)
            if isinstance(value, (dict, list, tuple, set, np.ndarray))
            else value
        )
    export_df.insert(0, "best_model_name", best_model_name)
    export_df.insert(0, "validation_group", validation_group)
    export_df.insert(0, "held_out_group", held_out_group)
    export_df.insert(0, "outer_fold", outer_fold)
    return export_df


def get_best_model_ancestors(
    leaderboard_df: pd.DataFrame,
    best_model_name: str,
) -> List[str]:
    if leaderboard_df.empty or "model" not in leaderboard_df.columns:
        return []

    matching_rows = leaderboard_df.loc[leaderboard_df["model"].astype(str) == best_model_name]
    if matching_rows.empty or "ancestors" not in matching_rows.columns:
        return []

    ancestors = matching_rows.iloc[0]["ancestors"]
    if isinstance(ancestors, str):
        return [ancestors]
    if isinstance(ancestors, (list, tuple, set, np.ndarray)):
        return [str(item) for item in ancestors]
    return []


def select_validation_group(
    global_group_order: Sequence[str],
    held_out_group: str,
) -> str:
    if held_out_group not in global_group_order:
        raise ValueError(f"Held-out group {held_out_group} was not found in the group order.")
    if len(global_group_order) < 2:
        raise ValueError("At least two activity-instance groups are required.")

    held_out_index = global_group_order.index(held_out_group)
    for offset in range(1, len(global_group_order)):
        candidate_group = global_group_order[(held_out_index + offset) % len(global_group_order)]
        if candidate_group != held_out_group:
            return candidate_group
    raise ValueError("Unable to select a validation activity-instance group.")


def train_autogluon_outer_fold(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: Sequence[str],
    output_dir: Path,
    fold_name: str,
    time_limit: int,
    ag_presets: str,
    ag_model_preset: str,
    random_state: int,
) -> Dict[str, object]:
    x_train = train_df[list(selected_features)].copy()
    y_train = train_df[LABEL_COLUMN].copy()
    x_validation = validation_df[list(selected_features)].copy()
    y_validation = validation_df[LABEL_COLUMN].copy()
    x_test = test_df[list(selected_features)].copy()
    x_train, x_validation, x_test = prepare_feature_matrices(
        x_train, x_validation, x_test
    )

    x_train, y_train = apply_smote_if_possible(x_train, y_train, random_state=random_state)
    if y_train.nunique() < 2:
        constant_label = str(y_train.iloc[0])
        return {
            "predictions": np.repeat(constant_label, len(x_test)),
            "probabilities": None,
            "best_model_name": "constant_predictor",
            "trained_model_names": ["constant_predictor"],
            "best_model_ancestors": [],
            "leaderboard": pd.DataFrame(),
        }
    if len(x_validation) == 0:
        raise ValueError("Validation dataframe is empty. A separate activity instance is required.")

    train_data = x_train.copy()
    train_data[LABEL_COLUMN] = y_train.to_numpy()
    tuning_data = x_validation.copy()
    tuning_data[LABEL_COLUMN] = y_validation.to_numpy()

    predictor_path = output_dir / "autogluon_runs" / fold_name
    if predictor_path.exists():
        shutil.rmtree(predictor_path, ignore_errors=True)
    hyperparameters = get_autogluon_hyperparameters(ag_model_preset)

    predictor = TabularPredictor(
        label=LABEL_COLUMN,
        path=str(predictor_path),
        eval_metric="balanced_accuracy",
        verbosity=0,
    )

    predictor.fit(
        train_data=train_data,
        tuning_data=tuning_data,
        presets=ag_presets,
        time_limit=time_limit,
        hyperparameters=hyperparameters,
        ag_args_fit={"num_gpus": 0},
    )
    predictions = predictor.predict(x_test).to_numpy()
    try:
        probabilities = predictor.predict_proba(x_test)
    except Exception:
        probabilities = None
    try:
        leaderboard_df = predictor.leaderboard(silent=True, extra_info=True)
    except TypeError:
        leaderboard_df = predictor.leaderboard(silent=True)
    best_model_name = predictor.model_best or "autogluon_best"
    if "model" in leaderboard_df.columns:
        trained_model_names = [str(model_name) for model_name in leaderboard_df["model"].tolist()]
    else:
        trained_model_names = [best_model_name]
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "best_model_name": best_model_name,
        "trained_model_names": trained_model_names,
        "best_model_ancestors": get_best_model_ancestors(leaderboard_df, best_model_name),
        "leaderboard": leaderboard_df,
    }


def run_nested_leave_one_activity_instance_out(
    feature_df: pd.DataFrame,
    output_dir: Path,
    top_k_grid: Sequence[int],
    time_limit: int,
    ag_presets: str,
    ag_model_preset: str,
    random_state: int,
    max_outer_folds: int | None,
    inner_max_groups: int | None,
) -> Dict[str, pd.DataFrame | Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "autogluon_runs").mkdir(parents=True, exist_ok=True)

    feature_columns = get_feature_columns(feature_df)
    group_splitter = LeaveOneGroupOut()
    groups = feature_df[GROUP_COLUMN].to_numpy()
    outer_folds = list(group_splitter.split(feature_df, groups=groups))
    if max_outer_folds is not None:
        outer_folds = outer_folds[:max_outer_folds]

    prediction_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    leaderboard_rows: List[pd.DataFrame] = []
    all_true: List[str] = []
    all_pred: List[str] = []
    selected_feature_count = int(len(feature_columns))
    global_group_order = sorted(feature_df[GROUP_COLUMN].astype(str).unique().tolist())

    print(f"Running {len(outer_folds)} outer folds...")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        train_df = feature_df.iloc[train_idx].reset_index(drop=True)
        test_df = feature_df.iloc[test_idx].reset_index(drop=True)
        held_out_group = str(test_df.iloc[0][GROUP_COLUMN])
        validation_group = select_validation_group(global_group_order, held_out_group)
        validation_mask = train_df[GROUP_COLUMN].astype(str).eq(validation_group)
        validation_df = train_df.loc[validation_mask].reset_index(drop=True)
        inner_train_df = train_df.loc[~validation_mask].reset_index(drop=True)
        validation_activity_instance_id = str(validation_df.iloc[0]["activity_instance_id"])
        print(
            f"  outer fold {fold_idx}/{len(outer_folds)}: "
            f"test={held_out_group}, validation={validation_group}"
        )

        selected_features = list(feature_columns)

        training_outputs = train_autogluon_outer_fold(
            train_df=inner_train_df,
            validation_df=validation_df,
            test_df=test_df,
            selected_features=selected_features,
            output_dir=output_dir,
            fold_name=f"outer_fold_{fold_idx:03d}",
            time_limit=time_limit,
            ag_presets=ag_presets,
            ag_model_preset=ag_model_preset,
            random_state=random_state,
        )
        y_pred = training_outputs["predictions"]
        y_prob = training_outputs["probabilities"]
        best_model_name = str(training_outputs["best_model_name"])
        trained_model_names = list(training_outputs["trained_model_names"])
        best_model_ancestors = list(training_outputs["best_model_ancestors"])
        leaderboard_rows.append(
            prepare_leaderboard_for_export(
                training_outputs["leaderboard"],
                outer_fold=fold_idx,
                held_out_group=held_out_group,
                validation_group=validation_group,
                best_model_name=best_model_name,
            )
        )

        y_true = test_df[LABEL_COLUMN].to_numpy()
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        fold_rows.append(
            {
                "outer_fold": fold_idx,
                "held_out_group": held_out_group,
                "validation_group": validation_group,
                "validation_activity_instance_id": validation_activity_instance_id,
                "n_train_segments": int(len(inner_train_df)),
                "n_validation_segments": int(len(validation_df)),
                "n_test_segments": int(len(test_df)),
                "selected_top_k": selected_feature_count,
                "ag_presets": ag_presets,
                "ag_model_preset": ag_model_preset,
                "best_model_name": best_model_name,
                "trained_model_names": json_dumps_compact(trained_model_names),
                "best_model_ancestors": json_dumps_compact(best_model_ancestors),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "weighted_f1": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
            }
        )
        validation_rows.append(
            {
                "outer_fold": fold_idx,
                "held_out_group": held_out_group,
                "held_out_activity_instance_id": str(test_df.iloc[0]["activity_instance_id"]),
                "validation_group": validation_group,
                "validation_activity_instance_id": validation_activity_instance_id,
                "validation_participant_id": str(validation_df.iloc[0]["participant_id"]),
                "n_train_groups": int(inner_train_df[GROUP_COLUMN].nunique()),
                "n_validation_segments": int(len(validation_df)),
                "n_test_segments": int(len(test_df)),
                "best_model_name": best_model_name,
                "trained_model_names": json_dumps_compact(trained_model_names),
                "best_model_ancestors": json_dumps_compact(best_model_ancestors),
            }
        )

        probability_columns = {}
        if y_prob is not None:
            for class_name in y_prob.columns:
                probability_columns[f"prob_{class_name}"] = y_prob[class_name].to_numpy()

        for row_idx, (_, test_row) in enumerate(test_df.iterrows()):
            row_record = {column: test_row[column] for column in METADATA_COLUMNS}
            row_record.update(
                {
                    "outer_fold": fold_idx,
                    "held_out_group": held_out_group,
                    "validation_group": validation_group,
                    "validation_activity_instance_id": validation_activity_instance_id,
                    "selected_top_k": selected_feature_count,
                    "best_model_name": best_model_name,
                    "predicted_pain_class": str(y_pred[row_idx]),
                    "correct": int(y_pred[row_idx] == y_true[row_idx]),
                }
            )
            for prob_column, prob_values in probability_columns.items():
                row_record[prob_column] = float(prob_values[row_idx])
            prediction_rows.append(row_record)

        checkpoint_outputs = {
            "predictions": pd.DataFrame(prediction_rows),
            "fold_metrics": pd.DataFrame(fold_rows),
            "validation_splits": pd.DataFrame(validation_rows),
            "model_leaderboard": pd.concat(leaderboard_rows, ignore_index=True) if leaderboard_rows else pd.DataFrame(),
            "inner_selection": pd.DataFrame(columns=["top_k", "mean_weighted_f1", "mean_macro_f1", "inner_folds", "outer_fold", "held_out_group"]),
            "summary_metrics": {
                "n_segments": int(len(feature_df)),
                "n_activity_instances": int(feature_df[GROUP_COLUMN].nunique()),
                "n_feature_columns": selected_feature_count,
                "outer_folds_completed": int(fold_idx),
                "outer_folds_total": int(len(outer_folds)),
                "feature_selection_mode": "all_features",
                "validation_selection_mode": "next_group_in_sorted_activity_instance_order",
                "selected_feature_count": selected_feature_count,
                "ag_presets": ag_presets,
                "ag_model_preset": ag_model_preset,
                "accuracy": float(accuracy_score(all_true, all_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
                "weighted_f1": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
                "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
                "class_order": CLASS_ORDER,
                "confusion_matrix": confusion_matrix(all_true, all_pred, labels=CLASS_ORDER).tolist(),
                "class_distribution": feature_df[LABEL_COLUMN].value_counts().sort_index().to_dict(),
            },
        }
        save_outputs(checkpoint_outputs, output_dir / "checkpoints")

    predictions_df = pd.DataFrame(prediction_rows)
    folds_df = pd.DataFrame(fold_rows)
    validation_df = pd.DataFrame(validation_rows)
    leaderboard_df = pd.concat(leaderboard_rows, ignore_index=True) if leaderboard_rows else pd.DataFrame()
    inner_df = pd.DataFrame(columns=["top_k", "mean_weighted_f1", "mean_macro_f1", "inner_folds", "outer_fold", "held_out_group"])

    confusion = confusion_matrix(all_true, all_pred, labels=CLASS_ORDER)
    metrics = {
        "n_segments": int(len(feature_df)),
        "n_activity_instances": int(feature_df[GROUP_COLUMN].nunique()),
        "n_feature_columns": selected_feature_count,
        "outer_folds_run": int(len(outer_folds)),
        "feature_selection_mode": "all_features",
        "validation_selection_mode": "next_group_in_sorted_activity_instance_order",
        "selected_feature_count": selected_feature_count,
        "ag_presets": ag_presets,
        "ag_model_preset": ag_model_preset,
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
        "weighted_f1": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        "class_order": CLASS_ORDER,
        "confusion_matrix": confusion.tolist(),
        "class_distribution": feature_df[LABEL_COLUMN].value_counts().sort_index().to_dict(),
    }

    return {
        "predictions": predictions_df,
        "fold_metrics": folds_df,
        "validation_splits": validation_df,
        "model_leaderboard": leaderboard_df,
        "inner_selection": inner_df,
        "summary_metrics": metrics,
    }


def save_outputs(
    outputs: Dict[str, pd.DataFrame | Dict[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "nested_logo_predictions.csv"
    folds_path = output_dir / "nested_logo_fold_metrics.csv"
    validation_path = output_dir / "nested_logo_validation_splits.csv"
    leaderboard_path = output_dir / "nested_logo_model_leaderboard.csv"
    inner_path = output_dir / "nested_logo_inner_selection.csv"
    metrics_path = output_dir / "nested_logo_summary_metrics.json"

    outputs["predictions"].to_csv(predictions_path, index=False)
    outputs["fold_metrics"].to_csv(folds_path, index=False)
    if "validation_splits" in outputs:
        outputs["validation_splits"].to_csv(validation_path, index=False)
    if "model_leaderboard" in outputs:
        outputs["model_leaderboard"].to_csv(leaderboard_path, index=False)
    outputs["inner_selection"].to_csv(inner_path, index=False)
    metrics_path.write_text(json.dumps(outputs["summary_metrics"], indent=2), encoding="utf-8")

    print(f"Saved predictions to {predictions_path}")
    print(f"Saved fold metrics to {folds_path}")
    if "validation_splits" in outputs:
        print(f"Saved validation splits to {validation_path}")
    if "model_leaderboard" in outputs:
        print(f"Saved model leaderboard to {leaderboard_path}")
    print(f"Saved inner-loop summary to {inner_path}")
    print(f"Saved overall metrics to {metrics_path}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_cache = (
        args.features_cache.resolve()
        if args.features_cache is not None
        else output_dir / "pain_features.csv"
    )

    feature_df = extract_feature_table(
        dataset_root=dataset_root,
        cache_path=features_cache,
        force=args.force_recompute_features,
    )

    print("Pain class distribution:")
    print(feature_df[LABEL_COLUMN].value_counts().sort_index().to_string())
    print(f"Healthy control files are excluded from training because they do not have pain labels.")

    if args.skip_cv:
        print("Feature extraction complete. Skipping nested CV as requested.")
        return

    outputs = run_nested_leave_one_activity_instance_out(
        feature_df=feature_df,
        output_dir=output_dir,
        top_k_grid=args.top_k_grid,
        time_limit=args.time_limit,
        ag_presets=args.ag_presets,
        ag_model_preset=args.ag_model_preset,
        random_state=args.random_state,
        max_outer_folds=args.max_outer_folds,
        inner_max_groups=args.inner_max_groups,
    )
    save_outputs(outputs, output_dir)

    print("Overall summary:")
    print(json.dumps(outputs["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
