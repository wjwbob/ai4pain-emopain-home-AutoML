from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from emopain_data_utils import (
    collect_skeleton_files,
    describe_supported_layouts,
    load_skeleton_array,
    parse_emopain_filename,
)


JOINT_NAMES: List[str] = [
    "hip",
    "mid_spine",
    "right_ankle",
    "right_elbow",
    "right_knee",
    "right_wrist",
]
PAIR_NAMES: List[Tuple[int, int]] = list(combinations(range(len(JOINT_NAMES)), 2))
AXES = ["x", "y", "z"]
WINDOW_STAT_NAMES = ["max", "min", "mean", "std", "median"]
META_COLUMNS = [
    "source_file",
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
    "window_duration_s",
    "step_duration_s",
    "window_size",
    "step_size",
    "num_windows",
    "torso_scale_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export all EmoPain@Home skeleton files (.txt or .npy) to one CSV with "
            "handcrafted movement features adapted from mediapipe_feature_extraction_new.py."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Dataset root containing the pain and healthy folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "emopain_all_npy_features.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Sliding window length in seconds. Default uses 1-second windows for variable frame rates.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=1.0,
        help="Sliding step length in seconds. Default uses a non-overlapping 1-second step.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Optional legacy override: fixed window size in frames. If set, this overrides --window-seconds.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Optional legacy override: fixed step size in frames. If set, this overrides --step-seconds.",
    )
    parser.add_argument(
        "--exclude-healthy",
        action="store_true",
        help="Only export pain cohort files.",
    )
    return parser.parse_args()


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def calculate_statistics(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "max": float("nan"),
            "min": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
        }
    return {
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def interpolate_missing_values(array_2d: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(array_2d)
    df = df.interpolate(method="linear", axis=0, limit_area="inside")
    # Keep edge gaps from propagating through every downstream feature.
    df = df.ffill().bfill()
    return df.to_numpy(dtype=np.float64)


def rowwise_nanmean(array_3d: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(array_3d)
    valid_counts = finite_mask.sum(axis=1)
    summed = np.nansum(array_3d, axis=1)
    return np.divide(
        summed,
        valid_counts,
        out=np.full_like(summed, np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )


def rowwise_reference_scale(coords: np.ndarray) -> np.ndarray:
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


def normalize_window(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    root = coords[:, 0, :]
    torso_distance = euclidean_distance(coords[:, 1, :], coords[:, 0, :])
    fallback_scale = rowwise_reference_scale(coords)
    torso_distance = np.where(
        np.isfinite(torso_distance) & (torso_distance >= 1e-8),
        torso_distance,
        fallback_scale,
    )
    torso_distance = np.where(
        np.isfinite(torso_distance) & (torso_distance >= 1e-8),
        torso_distance,
        np.nan,
    )
    normalized = (coords - root[:, None, :]) / torso_distance[:, None, None]
    return normalized, torso_distance


def calculate_kinematics(values: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(values, dt)
    acceleration = np.gradient(velocity, dt)
    jerk = np.gradient(acceleration, dt)
    return velocity, acceleration, jerk


def process_single_window(window_coords: np.ndarray, sampling_rate_hz: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    normalized, torso_distance = normalize_window(window_coords)
    center_point = rowwise_nanmean(normalized)

    distances: Dict[str, Dict[str, float]] = {
        "torso_scale": calculate_statistics(torso_distance),
    }

    for joint_a, joint_b in PAIR_NAMES:
        feature_name = f"dist_{JOINT_NAMES[joint_a]}__{JOINT_NAMES[joint_b]}"
        distances[feature_name] = calculate_statistics(
            euclidean_distance(normalized[:, joint_a, :], normalized[:, joint_b, :])
        )

    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        feature_name = f"dist_{joint_name}__body_center"
        distances[feature_name] = calculate_statistics(
            euclidean_distance(normalized[:, joint_idx, :], center_point)
        )

    kinematics: Dict[str, Dict[str, float]] = {}
    dt = 1.0 / sampling_rate_hz

    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        for axis_idx, axis_name in enumerate(AXES):
            axis_values = normalized[:, joint_idx, axis_idx]
            velocity, acceleration, jerk = calculate_kinematics(axis_values, dt=dt)

            kinematics[f"{joint_name}_{axis_name}_raw"] = calculate_statistics(axis_values)
            kinematics[f"{joint_name}_{axis_name}_velocity"] = calculate_statistics(velocity)
            kinematics[f"{joint_name}_{axis_name}_acceleration"] = calculate_statistics(acceleration)
            kinematics[f"{joint_name}_{axis_name}_jerk"] = calculate_statistics(jerk)

        magnitude = np.linalg.norm(normalized[:, joint_idx, :], axis=1)
        vel_mag = np.linalg.norm(np.gradient(normalized[:, joint_idx, :], dt, axis=0), axis=1)
        acc_mag = np.linalg.norm(np.gradient(np.gradient(normalized[:, joint_idx, :], dt, axis=0), dt, axis=0), axis=1)
        jerk_mag = np.linalg.norm(
            np.gradient(
                np.gradient(np.gradient(normalized[:, joint_idx, :], dt, axis=0), dt, axis=0),
                dt,
                axis=0,
            ),
            axis=1,
        )

        kinematics[f"{joint_name}_magnitude_raw"] = calculate_statistics(magnitude)
        kinematics[f"{joint_name}_magnitude_velocity"] = calculate_statistics(vel_mag)
        kinematics[f"{joint_name}_magnitude_acceleration"] = calculate_statistics(acc_mag)
        kinematics[f"{joint_name}_magnitude_jerk"] = calculate_statistics(jerk_mag)

    return {
        "distances": distances,
        "kinematics": kinematics,
    }


def calculate_meta_statistics(windows_stats: List[Dict[str, Dict[str, Dict[str, float]]]]) -> Dict[str, float]:
    meta_stats: Dict[str, float] = {}

    for feature_group in ["distances", "kinematics"]:
        feature_names = windows_stats[0][feature_group].keys()
        for feature_name in feature_names:
            for window_stat_name in WINDOW_STAT_NAMES:
                values = [
                    window[feature_group][feature_name][window_stat_name]
                    for window in windows_stats
                ]
                for aggregate_name, aggregate_fn in {
                    "max_of_all": np.max,
                    "min_of_all": np.min,
                    "mean_of_all": np.mean,
                    "std_of_all": np.std,
                    "median_of_all": np.median,
                }.items():
                    meta_stats[
                        f"{feature_name}_{aggregate_name}_{window_stat_name}"
                    ] = float(aggregate_fn(values))

    return meta_stats


def resolve_window_frames(
    sampling_rate_hz: float,
    window_seconds: float,
    step_seconds: float,
    window_size: int | None,
    step_size: int | None,
) -> Tuple[int, int]:
    if window_size is None:
        window_size = max(2, int(round(window_seconds * sampling_rate_hz)))
    else:
        window_size = max(2, int(window_size))

    if step_size is None:
        step_size = max(1, int(round(step_seconds * sampling_rate_hz)))
    else:
        step_size = max(1, int(step_size))

    return window_size, step_size


def process_file(
    path: Path,
    window_seconds: float,
    step_seconds: float,
    window_size: int | None,
    step_size: int | None,
) -> Dict[str, object]:
    metadata = parse_emopain_filename(path)
    raw = load_skeleton_array(path)
    if raw.ndim != 2 or raw.shape[1] != 18:
        raise ValueError(f"Expected shape (T, 18), got {raw.shape} for {path.name}")

    num_frames = int(raw.shape[0])
    nan_ratio = float(np.isnan(raw).sum() / raw.size)
    filled = interpolate_missing_values(raw)
    coords = filled.reshape(num_frames, len(JOINT_NAMES), 3)
    window_size_frames, step_size_frames = resolve_window_frames(
        sampling_rate_hz=float(metadata["sampling_rate_hz"]),
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        window_size=window_size,
        step_size=step_size,
    )

    if num_frames < window_size_frames:
        windows = [coords]
    else:
        windows = [
            coords[start_idx : start_idx + window_size_frames]
            for start_idx in range(0, num_frames - window_size_frames + 1, step_size_frames)
        ]

    if not windows:
        windows = [coords]

    windows_stats = [
        process_single_window(window, sampling_rate_hz=float(metadata["sampling_rate_hz"]))
        for window in windows
    ]

    row = dict(metadata)
    row.update(
        {
            "num_frames": num_frames,
            "nan_ratio": nan_ratio,
            "window_duration_s": float(window_seconds),
            "step_duration_s": float(step_seconds),
            "window_size": int(window_size_frames),
            "step_size": int(step_size_frames),
            "num_windows": len(windows_stats),
            "torso_scale_mean": float(
                np.mean([window["distances"]["torso_scale"]["mean"] for window in windows_stats])
            ),
        }
    )
    row.update(calculate_meta_statistics(windows_stats))
    return row
def export_all_features(
    dataset_root: Path,
    output_csv: Path,
    window_seconds: float,
    step_seconds: float,
    window_size: int | None,
    step_size: int | None,
    include_healthy: bool,
) -> pd.DataFrame:
    files = collect_skeleton_files(dataset_root, include_healthy=include_healthy)
    if not files:
        raise FileNotFoundError(
            "No supported skeleton files were found. "
            + describe_supported_layouts()
        )

    rows: List[Dict[str, object]] = []
    print(f"Processing {len(files)} skeleton files...")
    for idx, path in enumerate(files, start=1):
        if idx % 50 == 0 or idx == len(files):
            print(f"  processed {idx}/{len(files)} files")
        rows.append(
            process_file(
                path,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                window_size=window_size,
                step_size=step_size,
            )
        )

    df = pd.DataFrame(rows)
    metadata_cols = [col for col in META_COLUMNS if col in df.columns]
    feature_cols = sorted([col for col in df.columns if col not in metadata_cols])
    df = df[metadata_cols + feature_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved feature table to {output_csv}")
    return df


def main() -> None:
    args = parse_args()
    include_healthy = not args.exclude_healthy
    export_all_features(
        dataset_root=args.dataset_root.resolve(),
        output_csv=args.output_csv.resolve(),
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        window_size=args.window_size,
        step_size=args.step_size,
        include_healthy=include_healthy,
    )


if __name__ == "__main__":
    main()
