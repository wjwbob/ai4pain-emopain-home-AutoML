# Pereira Submission LOAO Audit

This note records exactly which parts of the Pereira submission are reused in
`evaluate_pereira_loao.py`, and which parts are intentionally replaced by the
local LOAO validation protocol.

## Reused From Pereira

- Input joint layout for EmoPain@Home pain files:
  six joints are read from 18 columns as contiguous XYZ triplets, then reordered
  to `x, z, y` exactly as in `pereira_submission_compact/preprocessing.py`.
- Feature extraction:
  - transpose `(6, T, 3)` to `(T, 6, 3)`
  - compute upper-triangular pairwise Euclidean distances between the 6 joints
  - replace missing pairwise distances with the per-frame mean distance
  - compute first derivative with `np.gradient(m, axis=0) * fps`
  - compute second derivative with `np.gradient(dm, axis=0) * fps`
  - concatenate distance, first derivative, and second derivative per frame
  - normalize the concatenated per-frame vector by its L2 norm
  - split back into the three matrices
  - summarize each matrix with mean, standard deviation, minimum, and maximum
- Feature dimensionality:
  15 pairwise distances * 3 signal groups * 4 statistics = 180 features.
- Training augmentation:
  Pereira-style `overlapped` mode builds 60-second pain windows with a
  15-second stride and uses the mean pain score within each window.
- Healthy PCA augmentation:
  fit PCA on non-overlapping 60-second healthy-control windows, use at most
  25 components, append `pca.transform(X)` to the 180 original features.
- Scaling:
  fit `MinMaxScaler` on the outer-fold training matrix after PCA augmentation,
  then transform train, validation, and test.
- Model:
  `sklearn.svm.SVR(C=2.0, kernel="rbf", gamma="scale")`.
- Sample weighting:
  inverse-frequency class weights computed from training labels with the same
  `len(y) / (3 * max(class_count, 1))` formula.
- Pain-class boundaries:
  low if pain `< 3`, medium if `3 <= pain <= 6`, high if pain `> 6`.
- Threshold optimization:
  grid search low/high thresholds from 2.0 to 7.0 in 0.1 increments, requiring
  a minimum gap of 0.5, maximizing the harmonic mean of balanced train and
  validation micro-F1.

## Intentionally Replaced For Local Challenge Evaluation

- Pereira's script uses the older EmoPain dataset as the validation set for
  threshold tuning. The LOAO evaluator replaces this with the challenge's local
  validation rule: the next sorted activity-instance group after the held-out
  test group.
- Pereira's script also includes healthy controls from the older EmoPain dataset
  in the PCA reference space. The LOAO evaluator uses only the healthy controls
  available under the provided EmoPain@Home dataset root, because the requested
  evaluation setup supplies only the challenge dataset and the LOAO protocol.
- Pereira's script imports `preprocessing.py`, which immediately loads or
  generates pickle caches. The LOAO evaluator avoids all participant-provided
  pickle files and reconstructs features from raw local data.

## Additional Control

`--training-mode segments` is provided as a stricter control that trains on the
original submitted pain segments only. The default `--training-mode overlapped`
is closer to Pereira's reported training pipeline.
