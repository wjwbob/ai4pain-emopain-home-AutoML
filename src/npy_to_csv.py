from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from emopain_data_utils import collect_skeleton_files, load_skeleton_array, parse_emopain_filename


DATASET_PATH = Path(__file__).resolve().parents[1]
SENSOR_NAMES = [
    "Hip",
    "Mid_spine",
    "Right_ankle",
    "Right_elbow",
    "Right_knee",
    "Right_wrist",
]


files = collect_skeleton_files(DATASET_PATH, include_healthy=True)
if not files:
    raise SystemExit("No supported skeleton files were found.")

random_file = random.choice(files)
metadata = parse_emopain_filename(random_file)
data = load_skeleton_array(random_file)
num_frames = data.shape[0]

print(f"Selected file: {random_file.name}")

feature_cols = [
    f"{sensor}_{coord}"
    for sensor in SENSOR_NAMES
    for coord in ("X", "Y", "Z")
]

df = pd.DataFrame(data, columns=feature_cols)
df.insert(0, "Time_Frame", np.arange(num_frames))
df["Source_File"] = str(metadata["source_file"])
df["Cohort"] = str(metadata["cohort"]).capitalize()
df["Participant_ID"] = str(metadata["participant_id"])
df["Activity_ID"] = int(metadata["activity_type_id"])
df["Activity_Instance_ID"] = int(metadata["activity_instance_id"])
df["Pain_Label"] = metadata["pain_score"]
df["Sampling_Rate"] = float(metadata["sampling_rate_hz"])
df["Index"] = metadata["segment_index"]

metadata_cols = [
    "Time_Frame",
    "Source_File",
    "Cohort",
    "Participant_ID",
    "Activity_ID",
    "Activity_Instance_ID",
    "Pain_Label",
    "Sampling_Rate",
    "Index",
]
df = df[metadata_cols + feature_cols]

output_csv = DATASET_PATH / "random_instance_data.csv"
df.to_csv(output_csv, index=False)
print(
    f"Successfully saved {num_frames} frames with metadata and "
    f"{len(feature_cols)} feature labels to {output_csv}"
)
