from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from emopain_data_utils import (
    collect_skeleton_files,
    describe_supported_layouts,
    load_skeleton_array,
    parse_emopain_filename,
)


CLASS_ORDER = ["LP", "MP", "HP"]
NUMERIC_TO_CLASS = np.array(CLASS_ORDER)
METADATA_COLUMNS = [
    "file_name",
    "cohort",
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
]
FEATURE_PREFIX = "pereira_f"
RANDOM_STATE = 42


@dataclass(frozen=True)
class PainSegment:
    path: Path
    metadata: Dict[str, object]
    xyz: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Pereira submission using its preprocessing/model family "
            "under the local LOAO validation protocol."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Dataset root containing EmoPainatHome_pain and EmoPain(at)Home_healthy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pereira_loao_outputs",
        help="Directory for extracted features, predictions, fold metrics, and summary JSON.",
    )
    parser.add_argument(
        "--features-cache",
        type=Path,
        default=None,
        help="Optional pain-segment feature cache. Defaults to <output-dir>/pereira_pain_features.csv.",
    )
    parser.add_argument(
        "--healthy-cache",
        type=Path,
        default=None,
        help="Optional healthy-reference feature cache. Defaults to <output-dir>/pereira_healthy_reference_features.csv.",
    )
    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
        help="Recompute Pereira features even if CSV caches already exist.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["overlapped", "segments"],
        default="overlapped",
        help=(
            "overlapped reproduces Pereira-style 60s windows with 15s stride for training; "
            "segments trains only on original submitted segments."
        ),
    )
    parser.add_argument(
        "--no-healthy-pca",
        action="store_true",
        help="Disable Pereira's healthy-control PCA feature augmentation.",
    )
    parser.add_argument(
        "--max-outer-folds",
        type=int,
        default=None,
        help="Optional fold limit for smoke tests.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
        help="Pereira training-window duration in seconds.",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=15.0,
        help="Pereira overlapped training-window stride in seconds.",
    )
    parser.add_argument("--threshold-low-bound", type=float, default=2.0)
    parser.add_argument("--threshold-high-bound", type=float, default=7.0)
    parser.add_argument("--threshold-step", type=float, default=0.1)
    parser.add_argument("--threshold-min-gap", type=float, default=0.5)
    return parser.parse_args()


def pain_to_numeric_class(y: Sequence[float]) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    return np.select([y_arr < 3, y_arr > 6], [0, 2], default=1).astype(int)


def numeric_to_class(y: Sequence[int]) -> np.ndarray:
    return NUMERIC_TO_CLASS[np.asarray(y, dtype=int)]


def pred_to_numeric_class_with_thresholds(
    y_pred: Sequence[float],
    thresholds: Sequence[float],
) -> np.ndarray:
    low_t, high_t = thresholds
    y_arr = np.asarray(y_pred, dtype=float)
    return np.select([y_arr < low_t, y_arr > high_t], [0, 2], default=1).astype(int)


def balanced_sample_weight(y: Sequence[int]) -> np.ndarray:
    y_arr = np.asarray(y, dtype=int)
    counts = np.bincount(y_arr, minlength=3)
    weights = len(y_arr) / (3 * np.maximum(counts, 1))
    return weights[y_arr]


def threshold_fitness(
    pred_train_raw: np.ndarray,
    y_train: np.ndarray,
    pred_val_raw: np.ndarray,
    y_val: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float, float]:
    yhat_train = pred_to_numeric_class_with_thresholds(pred_train_raw, thresholds)
    yhat_val = pred_to_numeric_class_with_thresholds(pred_val_raw, thresholds)
    train_sw = balanced_sample_weight(y_train)
    val_sw = balanced_sample_weight(y_val)
    f1_train = f1_score(
        y_train,
        yhat_train,
        average="micro",
        labels=[0, 1, 2],
        sample_weight=train_sw,
        zero_division=0,
    )
    f1_val = f1_score(
        y_val,
        yhat_val,
        average="micro",
        labels=[0, 1, 2],
        sample_weight=val_sw,
        zero_division=0,
    )
    fitness = (2 * f1_train * f1_val) / (f1_train + f1_val + 1e-12)
    return float(fitness), float(f1_train), float(f1_val)


def grid_search_thresholds(
    pred_train_raw: np.ndarray,
    y_train: np.ndarray,
    pred_val_raw: np.ndarray,
    y_val: np.ndarray,
    low_bound: float,
    high_bound: float,
    step: float,
    min_gap: float,
) -> Dict[str, object]:
    values = np.round(np.arange(low_bound, high_bound + 1e-9, step), 1)
    best_score = -np.inf
    best_thresholds: np.ndarray | None = None
    best_train_f1 = 0.0
    best_val_f1 = 0.0

    for low_t in values:
        for high_t in values:
            if high_t - low_t < min_gap:
                continue
            thresholds = np.array([low_t, high_t], dtype=float)
            score, train_f1, val_f1 = threshold_fitness(
                pred_train_raw,
                y_train,
                pred_val_raw,
                y_val,
                thresholds,
            )
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
                best_train_f1 = train_f1
                best_val_f1 = val_f1

    if best_thresholds is None:
        raise ValueError("Threshold grid search found no valid threshold pair.")
    return {
        "thresholds": best_thresholds,
        "score": float(best_score),
        "train_f1": float(best_train_f1),
        "val_f1": float(best_val_f1),
    }


def raw_to_pereira_xyz(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 18:
        raise ValueError(f"Expected a (T, 18) skeleton array, received {raw.shape}.")
    return np.stack([raw[:, i * 3 : (i + 1) * 3][:, [0, 2, 1]] for i in range(6)])


def pairwise_distances(data: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(data[:, :, None, :] - data[:, None, :, :], axis=-1)
    i, j = np.triu_indices(data.shape[1], k=1)
    dist = dist[:, i, j]
    row_mean = np.nanmean(dist, axis=1, keepdims=True)
    row_mean = np.nan_to_num(row_mean, nan=0.0)
    return np.where(np.isnan(dist), row_mean, dist)


def preprocess_pereira_features(xyz: np.ndarray, fps: float) -> np.ndarray:
    sequence = np.asarray(xyz, dtype=float).transpose(1, 0, 2)
    m = pairwise_distances(sequence)
    dm = np.gradient(m, axis=0) * float(fps)
    ddm = np.gradient(dm, axis=0) * float(fps)
    combined = np.concatenate([m, dm, ddm], axis=1)
    row_norm = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = np.divide(
        combined,
        row_norm,
        out=np.zeros_like(combined),
        where=row_norm > 0,
    )
    n_features = m.shape[1]
    matrices = [
        combined[:, :n_features],
        combined[:, n_features : 2 * n_features],
        combined[:, 2 * n_features :],
    ]
    return np.concatenate(
        [stat(mat, axis=0) for mat in matrices for stat in (np.mean, np.std, np.min, np.max)]
    )


def create_pereira_svr_regressor() -> SVR:
    return SVR(C=2.0, kernel="rbf", gamma="scale")


def metadata_from_path(path: Path) -> Dict[str, object]:
    parsed = parse_emopain_filename(path)
    participant_id = str(parsed["participant_id"])
    activity_instance_id = int(parsed["activity_instance_id"])
    group_id = f"{participant_id}_{activity_instance_id}"
    return {
        "file_name": path.name,
        "cohort": str(parsed["cohort"]),
        "participant_id": participant_id,
        "activity_type_id": int(parsed["activity_type_id"]),
        "activity_instance_id": activity_instance_id,
        "group_id": group_id,
        "pain_score": float(parsed["pain_score"])
        if parsed["pain_score"] is not None and np.isfinite(parsed["pain_score"])
        else np.nan,
        "pain_class": parsed["pain_class"],
        "sampling_rate_hz": float(parsed["sampling_rate_hz"]),
        "segment_index": int(parsed["segment_index"])
        if np.isfinite(parsed["segment_index"])
        else -1,
    }


def load_pain_segments(dataset_root: Path) -> List[PainSegment]:
    files = [
        path
        for path in collect_skeleton_files(dataset_root, include_healthy=False)
        if path.name.startswith("P")
    ]
    if not files:
        raise FileNotFoundError(
            "No pain skeleton files were found. " + describe_supported_layouts()
        )

    segments: List[PainSegment] = []
    for path in files:
        metadata = metadata_from_path(path)
        raw = load_skeleton_array(path)
        metadata["num_frames"] = int(raw.shape[0])
        metadata["nan_ratio"] = float(np.isnan(raw).sum() / raw.size)
        segments.append(
            PainSegment(path=path, metadata=metadata, xyz=raw_to_pereira_xyz(raw))
        )
    return sorted(
        segments,
        key=lambda item: (
            str(item.metadata["participant_id"]),
            int(item.metadata["activity_type_id"]),
            int(item.metadata["activity_instance_id"]),
            int(item.metadata["segment_index"]),
        ),
    )


def feature_columns(n_features: int) -> List[str]:
    return [f"{FEATURE_PREFIX}{idx:03d}" for idx in range(n_features)]


def build_pain_feature_table(segments: Sequence[PainSegment]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, segment in enumerate(segments, start=1):
        if idx % 50 == 0 or idx == len(segments):
            print(f"  extracted Pereira features for {idx}/{len(segments)} pain segments")
        features = preprocess_pereira_features(
            segment.xyz,
            float(segment.metadata["sampling_rate_hz"]),
        )
        row = dict(segment.metadata)
        row.update(
            {column: float(value) for column, value in zip(feature_columns(len(features)), features)}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_overlapped_training_table(
    segments: Sequence[PainSegment],
    window_seconds: float,
    stride_seconds: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped: Dict[str, List[PainSegment]] = {}
    for segment in segments:
        grouped.setdefault(str(segment.metadata["group_id"]), []).append(segment)

    for group_id, group_segments in grouped.items():
        group_segments = sorted(group_segments, key=lambda item: int(item.metadata["segment_index"]))
        fps = float(group_segments[0].metadata["sampling_rate_hz"])
        window_size = int(round(window_seconds * fps))
        stride_size = int(round(stride_seconds * fps))
        if window_size < 2 or stride_size < 1:
            continue

        full_xyz = np.concatenate([segment.xyz for segment in group_segments], axis=1)
        full_pain = np.concatenate(
            [
                np.full(segment.xyz.shape[1], float(segment.metadata["pain_score"]))
                for segment in group_segments
            ]
        )
        if full_xyz.shape[1] < window_size:
            continue

        first_meta = group_segments[0].metadata
        for window_idx, start in enumerate(
            range(0, full_xyz.shape[1] - window_size + 1, stride_size)
        ):
            end = start + window_size
            features = preprocess_pereira_features(full_xyz[:, start:end, :], fps)
            pain_score = float(np.mean(full_pain[start:end]))
            row = {
                "file_name": f"{group_id}_overlap_{window_idx:04d}",
                "cohort": "pain_overlapped",
                "participant_id": first_meta["participant_id"],
                "activity_type_id": first_meta["activity_type_id"],
                "activity_instance_id": first_meta["activity_instance_id"],
                "group_id": group_id,
                "pain_score": pain_score,
                "pain_class": numeric_to_class([pain_to_numeric_class([pain_score])[0]])[0],
                "sampling_rate_hz": fps,
                "segment_index": window_idx,
                "num_frames": window_size,
                "nan_ratio": float(np.isnan(full_xyz[:, start:end, :]).sum() / full_xyz[:, start:end, :].size),
            }
            row.update(
                {column: float(value) for column, value in zip(feature_columns(len(features)), features)}
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_healthy_reference_table(
    dataset_root: Path,
    window_seconds: float,
) -> pd.DataFrame:
    all_files = collect_skeleton_files(dataset_root, include_healthy=True)
    healthy_files = [path for path in all_files if path.name.startswith("H")]
    rows: List[Dict[str, object]] = []
    for path in healthy_files:
        metadata = metadata_from_path(path)
        raw = load_skeleton_array(path)
        xyz = raw_to_pereira_xyz(raw)
        fps = float(metadata["sampling_rate_hz"])
        window_size = int(round(window_seconds * fps))
        if window_size < 2 or xyz.shape[1] < window_size:
            continue
        for window_idx, start in enumerate(range(0, xyz.shape[1] - window_size + 1, window_size)):
            end = start + window_size
            features = preprocess_pereira_features(xyz[:, start:end, :], fps)
            row = dict(metadata)
            row["file_name"] = f"{path.name}::healthy_window_{window_idx:04d}"
            row["segment_index"] = window_idx
            row["num_frames"] = window_size
            row["nan_ratio"] = float(np.isnan(xyz[:, start:end, :]).sum() / xyz[:, start:end, :].size)
            row.update(
                {column: float(value) for column, value in zip(feature_columns(len(features)), features)}
            )
            rows.append(row)
    return pd.DataFrame(rows)


def load_or_extract_features(
    dataset_root: Path,
    pain_cache: Path,
    healthy_cache: Path,
    force: bool,
    window_seconds: float,
    stride_seconds: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pain_cache.exists() and healthy_cache.exists() and not force:
        print(f"Loading Pereira pain features from {pain_cache}")
        pain_df = pd.read_csv(pain_cache)
        overlapped_cache = pain_cache.with_name("pereira_overlapped_training_features.csv")
        if overlapped_cache.exists():
            print(f"Loading Pereira overlapped training features from {overlapped_cache}")
            overlapped_df = pd.read_csv(overlapped_cache)
        else:
            segments = load_pain_segments(dataset_root)
            overlapped_df = build_overlapped_training_table(
                segments,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
            )
            overlapped_df.to_csv(overlapped_cache, index=False)
        print(f"Loading Pereira healthy PCA features from {healthy_cache}")
        healthy_df = pd.read_csv(healthy_cache)
        return pain_df, overlapped_df, healthy_df

    print("Loading raw pain files and extracting Pereira features...")
    segments = load_pain_segments(dataset_root)
    pain_df = build_pain_feature_table(segments)
    overlapped_df = build_overlapped_training_table(
        segments,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )
    print("Extracting Pereira healthy-reference features...")
    healthy_df = build_healthy_reference_table(dataset_root, window_seconds=window_seconds)

    pain_cache.parent.mkdir(parents=True, exist_ok=True)
    pain_df.to_csv(pain_cache, index=False)
    overlapped_df.to_csv(
        pain_cache.with_name("pereira_overlapped_training_features.csv"),
        index=False,
    )
    healthy_df.to_csv(healthy_cache, index=False)
    print(f"Saved pain features to {pain_cache}")
    print(f"Saved overlapped training features to {pain_cache.with_name('pereira_overlapped_training_features.csv')}")
    print(f"Saved healthy PCA features to {healthy_cache}")
    return pain_df, overlapped_df, healthy_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in df.columns if column.startswith(FEATURE_PREFIX)]


def select_validation_group(global_group_order: Sequence[str], held_out_group: str) -> str:
    held_out_index = global_group_order.index(held_out_group)
    for offset in range(1, len(global_group_order)):
        candidate = global_group_order[(held_out_index + offset) % len(global_group_order)]
        if candidate != held_out_group:
            return candidate
    raise ValueError("Unable to select a validation group.")


def fit_transform_with_optional_healthy_pca(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    healthy_features: np.ndarray,
    use_healthy_pca: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    pca_info: Dict[str, object] = {
        "healthy_pca_enabled": bool(use_healthy_pca),
        "healthy_pca_components": 0,
        "healthy_pca_samples": int(len(healthy_features)),
        "healthy_pca_variance": None,
    }
    x_train = np.nan_to_num(np.asarray(x_train, dtype=float), nan=0.0)
    x_val = np.nan_to_num(np.asarray(x_val, dtype=float), nan=0.0)
    x_test = np.nan_to_num(np.asarray(x_test, dtype=float), nan=0.0)
    healthy_features = np.nan_to_num(np.asarray(healthy_features, dtype=float), nan=0.0)

    if use_healthy_pca and len(healthy_features) > 0:
        n_pca = min(25, healthy_features.shape[0], healthy_features.shape[1])
        if n_pca > 0:
            pca = PCA(n_components=n_pca, random_state=RANDOM_STATE).fit(healthy_features)
            x_train = np.concatenate([x_train, pca.transform(x_train)], axis=1)
            x_val = np.concatenate([x_val, pca.transform(x_val)], axis=1)
            x_test = np.concatenate([x_test, pca.transform(x_test)], axis=1)
            pca_info.update(
                {
                    "healthy_pca_components": int(n_pca),
                    "healthy_pca_variance": float(pca.explained_variance_ratio_.sum()),
                }
            )

    scaler = MinMaxScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test), pca_info


def run_pereira_loao(
    pain_df: pd.DataFrame,
    overlapped_df: pd.DataFrame,
    healthy_df: pd.DataFrame,
    output_dir: Path,
    training_mode: str,
    use_healthy_pca: bool,
    max_outer_folds: int | None,
    threshold_low_bound: float,
    threshold_high_bound: float,
    threshold_step: float,
    threshold_min_gap: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = get_feature_columns(pain_df)
    if not feature_cols:
        raise ValueError("No Pereira feature columns found.")

    group_splitter = LeaveOneGroupOut()
    groups = pain_df["group_id"].astype(str).to_numpy()
    outer_folds = list(group_splitter.split(pain_df, groups=groups))
    if max_outer_folds is not None:
        outer_folds = outer_folds[:max_outer_folds]

    global_group_order = sorted(pain_df["group_id"].astype(str).unique().tolist())
    healthy_features = (
        healthy_df[feature_cols].to_numpy(dtype=float)
        if not healthy_df.empty
        else np.empty((0, len(feature_cols)), dtype=float)
    )

    prediction_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    all_true: List[str] = []
    all_pred: List[str] = []

    train_source_df = overlapped_df if training_mode == "overlapped" else pain_df
    print(f"Running Pereira LOAO with {len(outer_folds)} outer folds...")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        train_groups = set(pain_df.iloc[train_idx]["group_id"].astype(str))
        test_df = pain_df.iloc[test_idx].reset_index(drop=True)
        held_out_group = str(test_df.iloc[0]["group_id"])
        validation_group = select_validation_group(global_group_order, held_out_group)

        train_mask = train_source_df["group_id"].astype(str).isin(train_groups)
        validation_mask = train_source_df["group_id"].astype(str).eq(validation_group)
        inner_train_df = train_source_df.loc[train_mask & ~validation_mask].reset_index(drop=True)
        validation_df = pain_df.loc[pain_df["group_id"].astype(str).eq(validation_group)].reset_index(drop=True)

        if inner_train_df.empty or validation_df.empty or test_df.empty:
            raise ValueError(f"Fold {fold_idx} has an empty train/validation/test split.")

        print(
            f"  outer fold {fold_idx}/{len(outer_folds)}: "
            f"test={held_out_group}, validation={validation_group}, "
            f"train_segments={len(inner_train_df)}"
        )

        x_train = inner_train_df[feature_cols].to_numpy(dtype=float)
        x_val = validation_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        x_train, x_val, x_test, pca_info = fit_transform_with_optional_healthy_pca(
            x_train,
            x_val,
            x_test,
            healthy_features,
            use_healthy_pca=use_healthy_pca,
        )

        y_train_raw = inner_train_df["pain_score"].to_numpy(dtype=float)
        y_val_raw = validation_df["pain_score"].to_numpy(dtype=float)
        y_test_raw = test_df["pain_score"].to_numpy(dtype=float)
        y_train = pain_to_numeric_class(y_train_raw)
        y_val = pain_to_numeric_class(y_val_raw)
        y_test = pain_to_numeric_class(y_test_raw)

        model = create_pereira_svr_regressor()
        model.fit(x_train, y_train_raw, sample_weight=balanced_sample_weight(y_train))

        pred_train_raw = np.clip(model.predict(x_train), 0, 10)
        pred_val_raw = np.clip(model.predict(x_val), 0, 10)
        pred_test_raw = np.clip(model.predict(x_test), 0, 10)

        search = grid_search_thresholds(
            pred_train_raw,
            y_train,
            pred_val_raw,
            y_val,
            low_bound=threshold_low_bound,
            high_bound=threshold_high_bound,
            step=threshold_step,
            min_gap=threshold_min_gap,
        )
        thresholds = np.asarray(search["thresholds"], dtype=float)
        pred_test_numeric = pred_to_numeric_class_with_thresholds(pred_test_raw, thresholds)
        y_true = numeric_to_class(y_test)
        y_pred = numeric_to_class(pred_test_numeric)

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        fold_rows.append(
            {
                "outer_fold": fold_idx,
                "held_out_group": held_out_group,
                "validation_group": validation_group,
                "validation_activity_instance_id": int(validation_df.iloc[0]["activity_instance_id"]),
                "n_train_segments": int(len(inner_train_df)),
                "n_validation_segments": int(len(validation_df)),
                "n_test_segments": int(len(test_df)),
                "training_mode": training_mode,
                "healthy_pca_enabled": bool(use_healthy_pca),
                "healthy_pca_components": int(pca_info["healthy_pca_components"]),
                "healthy_pca_variance": pca_info["healthy_pca_variance"],
                "low_threshold": float(thresholds[0]),
                "high_threshold": float(thresholds[1]),
                "threshold_fitness": float(search["score"]),
                "threshold_train_micro_f1": float(search["train_f1"]),
                "threshold_validation_micro_f1": float(search["val_f1"]),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }
        )
        validation_rows.append(
            {
                "outer_fold": fold_idx,
                "held_out_group": held_out_group,
                "held_out_activity_instance_id": int(test_df.iloc[0]["activity_instance_id"]),
                "validation_group": validation_group,
                "validation_activity_instance_id": int(validation_df.iloc[0]["activity_instance_id"]),
                "validation_participant_id": str(validation_df.iloc[0]["participant_id"]),
                "n_train_groups": int(inner_train_df["group_id"].nunique()),
                "n_validation_segments": int(len(validation_df)),
                "n_test_segments": int(len(test_df)),
            }
        )

        for row_idx, (_, test_row) in enumerate(test_df.iterrows()):
            row_record = {column: test_row[column] for column in METADATA_COLUMNS}
            row_record.update(
                {
                    "outer_fold": fold_idx,
                    "held_out_group": held_out_group,
                    "validation_group": validation_group,
                    "validation_activity_instance_id": int(validation_df.iloc[0]["activity_instance_id"]),
                    "training_mode": training_mode,
                    "low_threshold": float(thresholds[0]),
                    "high_threshold": float(thresholds[1]),
                    "predicted_pain_score": float(pred_test_raw[row_idx]),
                    "predicted_pain_class": str(y_pred[row_idx]),
                    "correct": int(y_pred[row_idx] == y_true[row_idx]),
                }
            )
            prediction_rows.append(row_record)

        checkpoint = build_outputs(
            pain_df,
            prediction_rows,
            fold_rows,
            validation_rows,
            all_true,
            all_pred,
            len(feature_cols),
            len(outer_folds),
            training_mode,
            use_healthy_pca,
        )
        save_outputs(checkpoint, output_dir / "checkpoints", suffix="_checkpoint")

    return build_outputs(
        pain_df,
        prediction_rows,
        fold_rows,
        validation_rows,
        all_true,
        all_pred,
        len(feature_cols),
        len(outer_folds),
        training_mode,
        use_healthy_pca,
    )


def build_outputs(
    pain_df: pd.DataFrame,
    prediction_rows: Sequence[Dict[str, object]],
    fold_rows: Sequence[Dict[str, object]],
    validation_rows: Sequence[Dict[str, object]],
    all_true: Sequence[str],
    all_pred: Sequence[str],
    n_feature_columns: int,
    outer_folds_total: int,
    training_mode: str,
    use_healthy_pca: bool,
) -> Dict[str, object]:
    confusion = confusion_matrix(all_true, all_pred, labels=CLASS_ORDER)
    return {
        "predictions": pd.DataFrame(prediction_rows),
        "fold_metrics": pd.DataFrame(fold_rows),
        "validation_splits": pd.DataFrame(validation_rows),
        "summary_metrics": {
            "n_segments": int(len(pain_df)),
            "n_activity_instances": int(pain_df["group_id"].nunique()),
            "n_feature_columns": int(n_feature_columns),
            "outer_folds_run": int(len(fold_rows)),
            "outer_folds_total": int(outer_folds_total),
            "model": "Pereira SVR(C=2.0, kernel='rbf', gamma='scale')",
            "feature_pipeline": "Pereira pairwise-distance statistics",
            "training_mode": training_mode,
            "validation_selection_mode": "next_group_in_sorted_activity_instance_order",
            "healthy_pca_enabled": bool(use_healthy_pca),
            "threshold_selection": "grid search on outer-fold train + LOAO validation group",
            "accuracy": float(accuracy_score(all_true, all_pred)) if all_true else None,
            "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)) if all_true else None,
            "weighted_f1": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)) if all_true else None,
            "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)) if all_true else None,
            "class_order": CLASS_ORDER,
            "confusion_matrix": confusion.tolist(),
            "class_distribution": pain_df["pain_class"].value_counts().sort_index().to_dict(),
        },
    }


def save_outputs(outputs: Dict[str, object], output_dir: Path, suffix: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs["predictions"].to_csv(output_dir / f"pereira_loao_predictions{suffix}.csv", index=False)
    outputs["fold_metrics"].to_csv(output_dir / f"pereira_loao_fold_metrics{suffix}.csv", index=False)
    outputs["validation_splits"].to_csv(output_dir / f"pereira_loao_validation_splits{suffix}.csv", index=False)
    (output_dir / f"pereira_loao_summary_metrics{suffix}.json").write_text(
        json.dumps(outputs["summary_metrics"], indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pain_cache = (
        args.features_cache.resolve()
        if args.features_cache is not None
        else output_dir / "pereira_pain_features.csv"
    )
    healthy_cache = (
        args.healthy_cache.resolve()
        if args.healthy_cache is not None
        else output_dir / "pereira_healthy_reference_features.csv"
    )

    pain_df, overlapped_df, healthy_df = load_or_extract_features(
        dataset_root=dataset_root,
        pain_cache=pain_cache,
        healthy_cache=healthy_cache,
        force=args.force_recompute_features,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    print("Pain class distribution:")
    print(pain_df["pain_class"].value_counts().sort_index().to_string())
    print(f"Pain segments: {len(pain_df)}")
    print(f"Overlapped training windows: {len(overlapped_df)}")
    print(f"Healthy PCA windows: {len(healthy_df)}")

    outputs = run_pereira_loao(
        pain_df=pain_df,
        overlapped_df=overlapped_df,
        healthy_df=healthy_df,
        output_dir=output_dir,
        training_mode=args.training_mode,
        use_healthy_pca=not args.no_healthy_pca,
        max_outer_folds=args.max_outer_folds,
        threshold_low_bound=args.threshold_low_bound,
        threshold_high_bound=args.threshold_high_bound,
        threshold_step=args.threshold_step,
        threshold_min_gap=args.threshold_min_gap,
    )
    save_outputs(outputs, output_dir)
    print("Overall summary:")
    print(json.dumps(outputs["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
