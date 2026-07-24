# Data setup

## Why the data are not in this repository

The EmoPain@Home recordings are research data distributed separately from this
codebase. They may be subject to access, consent, and redistribution conditions.
This repository therefore contains no raw recordings, participant-level feature
tables, or per-segment predictions.

Obtain the dataset through the AI4Pain 2026 challenge or the dataset's authorised
distribution channel, and follow the applicable terms of use.

## Expected local layout

Place the dataset directories in the repository root:

```text
ai4pain-emopain-movement-analysis/
├── EmoPainatHome_pain/
│   ├── P...txt
│   └── ...
├── EmoPain(at)Home_healthy/
│   ├── H...txt
│   └── ...
├── src/
└── ...
```

The AutoGluon baseline uses the chronic-pain cohort only. The healthy cohort is
not used for training or evaluation and is not treated as the low-pain class.

## Supported input

The data loader accepts comma-delimited `.txt` files and NumPy `.npy` arrays.
Pain filenames are expected to encode participant, activity, pain score,
sampling rate, and segment index following the challenge convention.

To keep accidental uploads unlikely, `.gitignore` excludes the known dataset
folders as well as common research-data and model-artifact formats.
