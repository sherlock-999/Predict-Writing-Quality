"""
CatBoost Regression: Predicting Essay Score from Writing Process Features
=========================================================================
Pipeline:
  1. Feature engineering  — v4_features (loads CSV cache if available)
  2. 5-seed × 10-fold stratified CV  — 50 models total for stable OOF
  3. CatBoost regression  — optimised with early stopping per fold
  4. Evaluation           — CV RMSE + per-fold scores, OOF averaged across seeds
  5. Feature importance   — PredictionValuesChange-based plot

Output:
  models/catboost_s{seed}_fold{fold}.cbm  — 50 trained models
  catboost_plots/01_oof_predictions.png
  catboost_plots/02_importance.png

Usage:
    conda run -n exp python train_catboost.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from v4_features import compute_features, FEATURE_COLS, CATEGORIES, CAT_PALETTE

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'catboost_model')
PLOT_DIR    = os.path.join(BASE_DIR, 'catboost_plots')
TFIDF_PATH  = os.path.join(MODELS_DIR, 'tfidf', 'tfidf_svd.pkl')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── CV config ─────────────────────────────────────────────────────────────────
N_FOLDS = 10
SEEDS   = [42, 21, 2022, 7, 4]   # diverse seeds for better ensemble

# ── Model parameters ──────────────────────────────────────────────────────────
# - loss_function: RMSE is directly minimised (regression standard)
# - iterations: upper bound; early stopping finds the true optimum
# - learning_rate: small value works well with early stopping
# - depth: max tree depth; CatBoost symmetric trees are less prone to overfit
#           than XGBoost/LGBM, so depth 6-8 is typically good
# - l2_leaf_reg: L2 regularisation on leaf values
# - subsample: row subsampling (requires bootstrap_type=Bernoulli)
# - colsample_bylevel: feature fraction per split level
# - early_stopping_rounds: stop if val RMSE hasn't improved for N rounds
CATBOOST_PARAMS = {
    'loss_function':        'RMSE',
    'eval_metric':          'RMSE',
    'iterations':           2000,
    'learning_rate':        0.05,
    'depth':                6,
    'l2_leaf_reg':          3.0,
    'bootstrap_type':       'Bernoulli',
    'subsample':            0.8,
    'colsample_bylevel':    0.8,
    'min_data_in_leaf':     20,
    'early_stopping_rounds': 50,
    'verbose':              0,
    'thread_count':         -1,    # use all CPU cores
}

# =============================================================================
# SECTION 1 – LOAD DATA
# =============================================================================
features_cache = os.path.join(DATA_DIR, 'train_features_v4.csv')

if os.path.exists(features_cache):
    print(f"\nLoading precomputed features from {features_cache}...")
    df = pd.read_csv(features_cache).fillna(0)
else:
    print("\nCache not found — computing features from raw logs...")
    print("  (run precompute_features.py first to speed this up)")
    logs   = pd.read_csv(os.path.join(DATA_DIR, 'train_logs.csv'))
    scores = pd.read_csv(os.path.join(DATA_DIR, 'train_scores.csv'))
    feat_df, tfidf_pipeline = compute_features(logs)
    df = feat_df.merge(scores, on='id').fillna(0)
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump(tfidf_pipeline, f)
    print(f"  Saved TF-IDF SVD pipeline → {TFIDF_PATH}")

print(f"  {df.shape[0]} essays × {len(FEATURE_COLS)} features")

X = df[FEATURE_COLS].values
y = df['score'].values

# =============================================================================
# SECTION 2 – REPEATED STRATIFIED K-FOLD CV
# =============================================================================
print(f"\n{'='*60}")
print(f"SECTION 2 – {len(SEEDS)}-SEED × {N_FOLDS}-FOLD CV  ({len(SEEDS) * N_FOLDS} models total)")
print(f"{'='*60}")

all_oof_preds      = np.zeros((len(SEEDS), len(y)))
importance_scores  = np.zeros(len(FEATURE_COLS))
n_models_saved     = 0

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n── Seed {seed} (seed {seed_idx + 1}/{len(SEEDS)}) ──")

    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y.astype(str)), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, feature_names=FEATURE_COLS)
        val_pool   = Pool(X_val,   y_val,   feature_names=FEATURE_COLS)

        params = {**CATBOOST_PARAMS, 'random_seed': seed}
        model  = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool)

        val_preds          = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        fold_rmse          = mean_squared_error(y_val, val_preds) ** 0.5

        importance_scores += model.get_feature_importance(train_pool)

        # Save model — predict_catboost.py globs catboost_s*_fold*.cbm
        model_path = os.path.join(MODELS_DIR, f'catboost_s{seed}_fold{fold}.cbm')
        model.save_model(model_path)
        n_models_saved += 1

        best_iter = model.get_best_iteration()
        print(f"  Fold {fold:2d} | best iter: {best_iter:4d} | val RMSE: {fold_rmse:.4f}")

    seed_oof_rmse = mean_squared_error(y, oof_preds) ** 0.5
    all_oof_preds[seed_idx] = oof_preds
    print(f"  → Seed {seed} OOF RMSE: {seed_oof_rmse:.4f}")

# =============================================================================
# SECTION 3 – EVALUATION
# =============================================================================
print(f"\n{'='*60}")
print("SECTION 3 – EVALUATION")
print(f"{'='*60}")

mean_oof_preds = all_oof_preds.mean(axis=0)
oof_rmse       = mean_squared_error(y, mean_oof_preds) ** 0.5
baseline_rmse  = mean_squared_error(y, np.full_like(y, y.mean())) ** 0.5

seed_rmses = [mean_squared_error(y, all_oof_preds[i]) ** 0.5 for i in range(len(SEEDS))]
print(f"\n  Per-seed OOF RMSE : {[f'{r:.4f}' for r in seed_rmses]}")
print(f"  Mean ± Std        : {np.mean(seed_rmses):.4f} ± {np.std(seed_rmses):.4f}")
print(f"  Ensemble OOF RMSE : {oof_rmse:.4f}  (avg of {len(SEEDS)} seed predictions)")
print(f"  Baseline RMSE     : {baseline_rmse:.4f}  (predicting mean score)")
print(f"  Improvement       : {baseline_rmse - oof_rmse:.4f}")
print(f"\n  Saved {n_models_saved} models to: {MODELS_DIR}/")

importance_scores /= (len(SEEDS) * N_FOLDS)

# =============================================================================
# SECTION 4 – OOF PREDICTION PLOTS
# =============================================================================
oof_df         = pd.DataFrame({'true': y, 'pred': mean_oof_preds})
oof_df['error'] = oof_df['pred'] - oof_df['true']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(oof_df['true'], oof_df['pred'], alpha=0.25, s=12, color='#2196F3')
axes[0].plot([0.5, 6], [0.5, 6], 'r--', linewidth=1.2, label='Perfect prediction')
axes[0].set_xlabel('True Score')
axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'OOF: True vs Predicted\nEnsemble RMSE = {oof_rmse:.4f}  '
                  f'({len(SEEDS)} seeds × {N_FOLDS} folds)')
axes[0].legend()

axes[1].hist(oof_df['error'], bins=40, color='#2196F3', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Prediction Error (pred − true)')
axes[1].set_ylabel('Count')
axes[1].set_title('OOF Residual Distribution')

plt.suptitle('Out-of-Fold Predictions — CatBoost', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_oof_predictions.png'), dpi=150)
plt.close()
print("\n[Saved] 01_oof_predictions.png")

# =============================================================================
# SECTION 5 – FEATURE IMPORTANCE
# =============================================================================
print(f"\n{'='*60}")
print("SECTION 5 – FEATURE IMPORTANCE")
print(f"{'='*60}")

imp_df = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': importance_scores,
}).sort_values('importance', ascending=False)

print(f"\nFeature importance (avg over {len(SEEDS) * N_FOLDS} models):")
print(imp_df.head(20).to_string(index=False))

legend_elements = [Patch(facecolor=c, label=k) for k, c in CAT_PALETTE.items()]
feat_arr = np.array(FEATURE_COLS)

order    = importance_scores.argsort()
sorted_f = feat_arr[order]
sorted_v = importance_scores[order]
colors   = [CAT_PALETTE[CATEGORIES[f]] for f in sorted_f]

fig, ax = plt.subplots(figsize=(10, max(8, len(FEATURE_COLS) * 0.18)))
bars = ax.barh(sorted_f, sorted_v, color=colors, edgecolor='white')
ax.set_xlabel('Average PredictionValuesChange Importance', fontsize=11)
ax.set_title(
    f'Feature Importance (avg over {len(SEEDS) * N_FOLDS} models) — CatBoost',
    fontsize=12, fontweight='bold',
)
total = sorted_v.sum()
for bar, v in zip(bars, sorted_v):
    if total > 0:
        ax.text(v + total * 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v / total * 100:.1f}%', va='center', ha='left', fontsize=7)
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Category')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_importance.png'), dpi=150)
plt.close()
print("[Saved] 02_importance.png")

print(f"\nAll plots saved to: {PLOT_DIR}/")
print("Done.")
