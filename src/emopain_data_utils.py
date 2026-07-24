from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


PAIN_DIR_CANDIDATES: Sequence[str] = (
    "EmoPainatHome_pain",
)
HEALTHY_DIR_CANDIDATES: Sequence[str] = (
    "EmoPain(at)Home_healthy",
)
SUPPORTED_DATA_EXTENSIONS: Sequence[str] = (".txt", ".npy")


def pain_to_class(pain_score: float | None) -> str | None:
    if pain_score is None or np.isnan(pain_score):
        return None
    if pain_score < 3:
        return "LP"
    if pain_score <= 6:
        return "MP"
    return "HP"


def _parse_decimal_token(token: str) -> float:
    normalized = token
    if "." not in normalized and normalized.count("-") == 1:
        left, right = normalized.split("-")
        if left.isdigit() and right.isdigit():
            normalized = f"{left}.{right}"
    return float(normalized)


def resolve_dataset_dir(
    dataset_root: Path,
    candidates: Sequence[str],
) -> Path | None:
    for candidate in candidates:
        candidate_path = dataset_root / candidate
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path
    return None


def list_supported_files(directory: Path) -> List[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
    )


def collect_skeleton_files(dataset_root: Path, include_healthy: bool) -> List[Path]:
    files: List[Path] = []

    pain_dir = resolve_dataset_dir(dataset_root, PAIN_DIR_CANDIDATES)
    if pain_dir is not None:
        files.extend(list_supported_files(pain_dir))

    if include_healthy:
        healthy_dir = resolve_dataset_dir(dataset_root, HEALTHY_DIR_CANDIDATES)
        if healthy_dir is not None:
            files.extend(list_supported_files(healthy_dir))

    return sorted(files)


def load_skeleton_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path)
    elif suffix == ".txt":
        raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    else:
        raise ValueError(f"Unsupported file extension for skeleton data: {path.name}")

    return np.atleast_2d(np.asarray(raw, dtype=np.float64))


def parse_emopain_filename(path: Path) -> Dict[str, object]:
    parts = path.stem.split("_")

    if path.stem.startswith("P"):
        if len(parts) != 6:
            raise ValueError(f"Unexpected pain filename format: {path.name}")
        pain_score = _parse_decimal_token(parts[3])
        return {
            "source_file": path.name,
            "cohort": "pain",
            "participant_id": parts[0],
            "activity_type_id": int(parts[1]),
            "activity_instance_id": int(parts[2]),
            "group_id": f"{parts[0]}_{parts[1]}_{parts[2]}",
            "pain_score": pain_score,
            "pain_class": pain_to_class(pain_score),
            "sampling_rate_hz": float(parts[4]),
            "segment_index": int(parts[5]),
        }

    if path.stem.startswith("H"):
        if len(parts) not in {4, 5}:
            raise ValueError(f"Unexpected healthy filename format: {path.name}")
        segment_index = int(parts[4]) if len(parts) == 5 else np.nan
        return {
            "source_file": path.name,
            "cohort": "healthy",
            "participant_id": parts[0],
            "activity_type_id": int(parts[1]),
            "activity_instance_id": int(parts[2]),
            "group_id": f"{parts[0]}_{parts[1]}_{parts[2]}",
            "pain_score": np.nan,
            "pain_class": None,
            "sampling_rate_hz": float(parts[3]),
            "segment_index": segment_index,
        }

    raise ValueError(f"Unexpected filename prefix: {path.name}")


def describe_supported_layouts() -> str:
    return (
        f"pain directories={list(PAIN_DIR_CANDIDATES)}, "
        f"healthy directories={list(HEALTHY_DIR_CANDIDATES)}, "
        f"extensions={list(SUPPORTED_DATA_EXTENSIONS)}"
    )
