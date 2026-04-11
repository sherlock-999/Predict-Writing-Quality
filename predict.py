"""
Inference: Predict Essay Scores from test_logs.csv
====================================================
Loads the 5 fold models saved by train_lgbm.py, applies identical feature
engineering to test_logs.csv, averages predictions across folds, and writes
a submission.csv matching the format of sample_submission.csv.

Usage:
    conda run -n exp python predict.py

Output:
    writing_process/submission.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import compute_features, FEATURE_COLS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUT_PATH = os.path.join(BASE_DIR, 'submission.csv')

# =============================================================================
# SECTION 1 – LOAD MODELS
# =============================================================================
model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, 'lgbm_fold*.txt')))
if not model_paths:
    raise FileNotFoundError(
        f"No fold models found in {MODELS_DIR}/\n"
        "Run train_lgbm.py first to train and save the models."
    )

models = [lgb.Booster(model_file=p) for p in model_paths]
print(f"Loaded {len(models)} fold models from {MODELS_DIR}/")

# =============================================================================
# SECTION 2 – FEATURE ENGINEERING
# =============================================================================
print("Loading test_logs.csv...")
logs = pd.read_csv(os.path.join(DATA_DIR, 'test_logs.csv'))
print(f"  {logs.shape[0]:,} rows | {logs['id'].nunique()} essays")

print("Computing per-essay features...")
test_df = compute_features(logs).fillna(0)
X_test  = test_df[FEATURE_COLS].values
print(f"  Feature matrix: {X_test.shape[0]} essays × {X_test.shape[1]} features")

# =============================================================================
# SECTION 3 – INFERENCE
# =============================================================================
# Average predictions from all 5 fold models (ensemble).
# Each model was trained on 80% of the data; averaging reduces variance.
print("\nRunning inference...")
fold_preds = np.stack([model.predict(X_test) for model in models], axis=0)
preds      = fold_preds.mean(axis=0)

# Clip to the valid score range seen during training [0.5, 6.0].
# LightGBM can occasionally extrapolate slightly outside the training range.
preds = np.clip(preds, 0.5, 6.0)

print(f"  Prediction stats — mean: {preds.mean():.3f} | "
      f"std: {preds.std():.3f} | min: {preds.min():.3f} | max: {preds.max():.3f}")

# =============================================================================
# SECTION 4 – WRITE SUBMISSION
# =============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'score': preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to: {OUTPUT_PATH}")
print(f"  {len(submission)} rows")
print(submission.head(10).to_string(index=False))
