"""
Ensemble Inference: LightGBM + XGBoost → submission.csv
=========================================================
Loads all 5 LightGBM fold models and all 5 XGBoost fold models,
averages predictions within each family, then blends the two families
using inverse-OOF-RMSE weighting (better model gets more weight).

Usage:
    conda run -n exp python predict_ensemble.py

Requires:
    models/lgbm_fold{1..5}.txt   — saved by train_lgbm.py
    models/xgb_fold{1..5}.json   — saved by train_xgb.py

Output:
    submission_ensemble.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from important_features import compute_features, FEATURE_COLS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUT_PATH = os.path.join(BASE_DIR, 'submission_ensemble.csv')

# OOF RMSEs from training — paste the values printed by each train script.
# These control the blend weight: lower RMSE → higher weight.
# If you haven't run training yet, set both to 1.0 for a simple average.
OOF_RMSE_LGBM = 0.6  # replace with the OOF RMSE printed by train_lgbm.py
OOF_RMSE_XGB  = 0.6  # replace with the OOF RMSE printed by train_xgb.py

# =============================================================================
# SECTION 1 – LOAD MODELS
# =============================================================================
lgbm_paths = sorted(glob.glob(os.path.join(MODELS_DIR, 'lgbm_fold*.txt')))
xgb_paths  = sorted(glob.glob(os.path.join(MODELS_DIR, 'xgb_fold*.json')))

if not lgbm_paths:
    raise FileNotFoundError(f"No LightGBM models found in {MODELS_DIR}/. Run train_lgbm.py first.")
if not xgb_paths:
    raise FileNotFoundError(f"No XGBoost models found in {MODELS_DIR}/. Run train_xgb.py first.")

lgbm_models = [lgb.Booster(model_file=p) for p in lgbm_paths]
xgb_models  = [xgb.Booster(model_file=p) for p in xgb_paths]

print(f"Loaded {len(lgbm_models)} LightGBM models, {len(xgb_models)} XGBoost models")

# =============================================================================
# SECTION 2 – FEATURE ENGINEERING
# =============================================================================
print("\nLoading test_logs.csv...")
logs = pd.read_csv(os.path.join(DATA_DIR, 'test_logs.csv'))
print(f"  {logs.shape[0]:,} rows | {logs['id'].nunique()} essays")

print("Computing per-essay features...")
test_df = compute_features(logs).fillna(0)
X_test  = test_df[FEATURE_COLS].values
print(f"  Feature matrix: {X_test.shape[0]} essays × {X_test.shape[1]} features")

# XGBoost Booster expects DMatrix input
dtest = xgb.DMatrix(X_test, feature_names=[f'f{i}' for i in range(X_test.shape[1])])

# =============================================================================
# SECTION 3 – PREDICT PER MODEL FAMILY
# =============================================================================
lgbm_preds = np.stack([m.predict(X_test) for m in lgbm_models], axis=0).mean(axis=0)
xgb_preds  = np.stack([m.predict(dtest)  for m in xgb_models],  axis=0).mean(axis=0)

print(f"\nLightGBM  — mean: {lgbm_preds.mean():.3f} | std: {lgbm_preds.std():.3f}")
print(f"XGBoost   — mean: {xgb_preds.mean():.3f}  | std: {xgb_preds.std():.3f}")

# =============================================================================
# SECTION 4 – BLEND
# =============================================================================
# Inverse-RMSE weighting: the model with lower OOF error gets a larger weight.
# If both RMSEs are equal this reduces to a simple 50/50 average.
w_lgbm = 1.0 / OOF_RMSE_LGBM
w_xgb  = 1.0 / OOF_RMSE_XGB
w_total = w_lgbm + w_xgb

preds = (w_lgbm * lgbm_preds + w_xgb * xgb_preds) / w_total
preds = np.clip(preds, 0.5, 6.0)

print(f"\nBlend weights — LightGBM: {w_lgbm/w_total:.3f} | XGBoost: {w_xgb/w_total:.3f}")
print(f"Ensemble      — mean: {preds.mean():.3f} | std: {preds.std():.3f} "
      f"| min: {preds.min():.3f} | max: {preds.max():.3f}")

# =============================================================================
# SECTION 5 – WRITE SUBMISSION
# =============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'score': preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to: {OUTPUT_PATH}")
print(f"  {len(submission)} rows")
print(submission.head(10).to_string(index=False))
