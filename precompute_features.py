"""
Precompute Features and Export to CSV
======================================
Run this once to compute all per-essay features from the raw keystroke logs
and save them to data/train_features_v4.csv and data/test_features_v4.csv.

The fitted TF-IDF SVD pipeline is saved to models/tfidf_svd.pkl so that
train and test use the *same* transformer (preventing leakage and component
misalignment between sets).

train_lgbm.py and tune_lgbm.py can then load the CSV directly instead of
recomputing features on every run.

Usage:
    conda run -n exp python precompute_features.py
"""

import os
import pickle
import pandas as pd
from v6_features import compute_features

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'tfidf_updated')
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# TRAIN  — fit TF-IDF SVD on training corpus
# =============================================================================
print("Loading train_logs.csv...")
train_logs   = pd.read_csv(os.path.join(DATA_DIR, 'train_logs.csv'))
train_scores = pd.read_csv(os.path.join(DATA_DIR, 'train_scores.csv'))
print(f"  {train_logs.shape[0]:,} rows | {train_logs['id'].nunique()} essays")

print("Computing train features (fits TF-IDF SVD on training corpus)...")
train_feat_df, tfidf_pipeline = compute_features(train_logs)
train_df = train_feat_df.merge(train_scores, on='id').fillna(0)

out_path = os.path.join(DATA_DIR, 'train_features_v6.csv')
train_df.to_csv(out_path, index=False)
print(f"  Saved {train_df.shape[0]} essays × {train_df.shape[1]} columns → {out_path}")

# Save the fitted TF-IDF SVD pipeline so test features use the same transformer
tfidf_path = os.path.join(MODELS_DIR, 'tfidf_svd.pkl')
with open(tfidf_path, 'wb') as f:
    pickle.dump(tfidf_pipeline, f)
print(f"  Saved TF-IDF SVD pipeline → {tfidf_path}")

# =============================================================================
# TEST  — transform using the already-fitted TF-IDF SVD
# =============================================================================
print("\nLoading test_logs.csv...")
test_logs = pd.read_csv(os.path.join(DATA_DIR, 'test_logs.csv'))
print(f"  {test_logs.shape[0]:,} rows | {test_logs['id'].nunique()} essays")

print("Computing test features (applying fitted TF-IDF SVD)...")
test_feat_df, _ = compute_features(test_logs, tfidf_pipeline=tfidf_pipeline)
test_df = test_feat_df.fillna(0)

out_path = os.path.join(DATA_DIR, 'test_features_v6.csv')
test_df.to_csv(out_path, index=False)
print(f"  Saved {test_df.shape[0]} essays × {test_df.shape[1]} columns → {out_path}")

print("\nDone. Re-run this script only if features change.")
