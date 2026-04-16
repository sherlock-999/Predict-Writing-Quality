"""
Precompute v6 Features and Export to CSV
=========================================
Run this once to compute all per-essay features from the raw keystroke logs
and save them to data/train_features_v6.csv and data/test_features_v6.csv.

Two TF-IDF SVD pipelines are fitted on the training corpus and saved:
  tfidf/tfidf_svd.pkl       — char 2–4-gram on reconstructed essay text
  tfidf/event_tfidf_svd.pkl — word 1–5-gram on down_event sequence (v6 new)

Both pipelines are shared across all model types at train and inference time.

Usage:
    conda run -n exp python precompute_features_v6.py
"""

import os
import pickle
import pandas as pd
from v6_features import compute_features

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
TFIDF_DIR  = os.path.join(BASE_DIR, 'tfidf_v6')
os.makedirs(TFIDF_DIR, exist_ok=True)

TFIDF_PATH       = os.path.join(TFIDF_DIR, 'tfidf_svd.pkl')
EVENT_TFIDF_PATH = os.path.join(TFIDF_DIR, 'event_tfidf_svd.pkl')

# =============================================================================
# TRAIN  — fit both TF-IDF SVD pipelines on training corpus
# =============================================================================
print("Loading train_logs.csv...")
train_logs   = pd.read_csv(os.path.join(DATA_DIR, 'train_logs.csv'))
train_scores = pd.read_csv(os.path.join(DATA_DIR, 'train_scores.csv'))
print(f"  {train_logs.shape[0]:,} rows | {train_logs['id'].nunique()} essays")

print("Computing train features (fits both TF-IDF SVD pipelines on training corpus)...")
train_feat_df, tfidf_pipeline, event_tfidf_pipeline = compute_features(train_logs)
train_df = train_feat_df.merge(train_scores, on='id').fillna(0)

out_path = os.path.join(DATA_DIR, 'train_features_v6.csv')
train_df.to_csv(out_path, index=False)
print(f"  Saved {train_df.shape[0]} essays × {train_df.shape[1]} columns → {out_path}")

# Save both fitted pipelines
with open(TFIDF_PATH, 'wb') as f:
    pickle.dump(tfidf_pipeline, f)
print(f"  Saved text TF-IDF SVD pipeline       → {TFIDF_PATH}")

with open(EVENT_TFIDF_PATH, 'wb') as f:
    pickle.dump(event_tfidf_pipeline, f)
print(f"  Saved event TF-IDF SVD pipeline      → {EVENT_TFIDF_PATH}")

# =============================================================================
# TEST  — transform using the already-fitted pipelines
# =============================================================================
print("\nLoading test_logs.csv...")
test_logs = pd.read_csv(os.path.join(DATA_DIR, 'test_logs.csv'))
print(f"  {test_logs.shape[0]:,} rows | {test_logs['id'].nunique()} essays")

print("Computing test features (applying fitted TF-IDF SVD pipelines)...")
test_feat_df, _, _ = compute_features(
    test_logs,
    tfidf_pipeline=tfidf_pipeline,
    event_tfidf_pipeline=event_tfidf_pipeline,
)
test_df = test_feat_df.fillna(0)

out_path = os.path.join(DATA_DIR, 'test_features_v6.csv')
test_df.to_csv(out_path, index=False)
print(f"  Saved {test_df.shape[0]} essays × {test_df.shape[1]} columns → {out_path}")

print("\nDone. Re-run this script only if features change.")
