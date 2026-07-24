from pathlib import Path

import numpy as np
import pytest

from emopain_data_utils import pain_to_class, parse_emopain_filename


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "LP"),
        (2.99, "LP"),
        (3.0, "MP"),
        (6.0, "MP"),
        (6.01, "HP"),
        (np.nan, None),
        (None, None),
    ],
)
def test_pain_to_class_boundaries(score, expected):
    assert pain_to_class(score) == expected


def test_parse_pain_filename():
    metadata = parse_emopain_filename(Path("P213_8_34_4-0_40_323.txt"))

    assert metadata["participant_id"] == "P213"
    assert metadata["activity_type_id"] == 8
    assert metadata["activity_instance_id"] == 34
    assert metadata["group_id"] == "P213_8_34"
    assert metadata["pain_score"] == 4.0
    assert metadata["pain_class"] == "MP"
    assert metadata["sampling_rate_hz"] == 40.0
    assert metadata["segment_index"] == 323


def test_parse_healthy_filename():
    metadata = parse_emopain_filename(Path("H001_2_3_40_9.npy"))

    assert metadata["cohort"] == "healthy"
    assert metadata["participant_id"] == "H001"
    assert metadata["group_id"] == "H001_2_3"
    assert metadata["pain_score"] is np.nan or np.isnan(metadata["pain_score"])
    assert metadata["pain_class"] is None
    assert metadata["sampling_rate_hz"] == 40.0
    assert metadata["segment_index"] == 9
