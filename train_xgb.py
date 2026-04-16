"""
XGBoost Regression: Predicting Essay Score from Writing Process Features
========================================================================
Pipeline:
  1. Feature engineering  — testing_new_features (loads CSV cache if available)
  2. 5-seed × 10-fold stratified CV  — 50 models total for stable OOF
  3. XGBoost regression    — optimised with early stopping per fold
  4. Evaluation            — CV RMSE + per-fold scores, OOF averaged across seeds
  5. Feature importance    — gain-based and weight-based plots

Output:
  models/xgb_s{seed}_fold{fold}.json  — 50 trained models
  xgb_plots/01_oof_predictions.png
  xgb_plots/02_importance_gain.png
  xgb_plots/03_importance_weight.png

Usage:
    conda run -n exp python train_xgb.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from v6_features import compute_features, FEATURE_COLS, CATEGORIES, CAT_PALETTE

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'xgb_model')
PLOT_DIR    = os.path.join(BASE_DIR, 'xgb_plots')
TFIDF_DIR        = os.path.join(MODELS_DIR, 'tfidf_v6')
TFIDF_PATH       = os.path.join(TFIDF_DIR, 'tfidf_svd.pkl')
EVENT_TFIDF_PATH = os.path.join(TFIDF_DIR, 'event_tfidf_svd.pkl')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── CV config ─────────────────────────────────────────────────────────────────
N_FOLDS = 10
SEEDS   = [42, 21, 2022, 7, 4]   # diverse seeds for better ensemble

# =============================================================================
# SECTION 1 – LOAD DATA
# =============================================================================
features_cache = os.path.join(DATA_DIR, 'train_features_v6.csv')

if os.path.exists(features_cache):
    print(f"\nLoading precomputed features from {features_cache}...")
    df = pd.read_csv(features_cache).fillna(0)
else:
    print("\nCache not found — computing features from raw logs...")
    print("  (run precompute_features.py first to speed this up)")
    logs   = pd.read_csv(os.path.join(DATA_DIR, 'train_logs.csv'))
    scores = pd.read_csv(os.path.join(DATA_DIR, 'train_scores.csv'))
    feat_df, tfidf_pipeline, event_tfidf_pipeline = compute_features(logs)
    df = feat_df.merge(scores, on='id').fillna(0)
    os.makedirs(TFIDF_DIR, exist_ok=True)
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump(tfidf_pipeline, f)
    print(f"  Saved text TF-IDF SVD pipeline  → {TFIDF_PATH}")
    with open(EVENT_TFIDF_PATH, 'wb') as f:
        pickle.dump(event_tfidf_pipeline, f)
    print(f"  Saved event TF-IDF SVD pipeline → {EVENT_TFIDF_PATH}")

print(f"  {df.shape[0]} essays × {len(FEATURE_COLS)} features")

X = df[FEATURE_COLS].values
y = df['score'].values

# =============================================================================
# SECTION 2 – MODEL CONFIGURATION
# =============================================================================

# XGBoost parameters.
# - objective='reg:squarederror' : minimise MSE (standard regression)
# - eval_metric='rmse'           : report RMSE on validation set
# - max_depth=6                  : max tree depth; shallower = more regularised
# - min_child_weight=5           : min sum of instance weights in a child node
# - learning_rate=0.05           : small steps so early stopping finds the optimum
# - subsample=0.8                : row-level subsampling — reduces overfitting
# - colsample_bytree=0.8         : fraction of features per tree — decorrelates trees
# - reg_alpha=0.1                : L1 regularisation on leaf weights
# - reg_lambda=1.0               : L2 regularisation on leaf weights
# - n_estimators=2000            : upper bound; early stopping finds true optimum
# - early_stopping_rounds=50     : stop if val RMSE hasn't improved for 50 rounds
XGB_PARAMS = {
    'learning_rate': 0.005,
    'objective' : 'reg:squarederror',
    'reg_alpha': 0.0008774661176012108,
    'reg_lambda': 2.542812743920178,
    'colsample_bynode': 0.7839026197349153,
    'subsample': 0.8994226268096415, 
    'eta': 0.04730766698056879, 
    'tree_method': "gpu_hist",
    'n_estimators': 2000,
    'random_state': 42,
    'eval_metric': 'rmse',
    'device':             'cuda',
    'early_stopping_rounds': 50,
}

# =============================================================================
# SECTION 3 – REPEATED STRATIFIED K-FOLD CV
# =============================================================================
print(f"\n{'='*60}")
print(f"SECTION 3 – {len(SEEDS)}-SEED × {N_FOLDS}-FOLD CV  ({len(SEEDS) * N_FOLDS} models total)")
print(f"{'='*60}")

# Accumulate OOF predictions across seeds then average for final RMSE.
all_oof_preds = np.zeros((len(SEEDS), len(y)))

# Accumulate feature importances across all models.
importance_gain   = np.zeros(len(FEATURE_COLS))
importance_weight = np.zeros(len(FEATURE_COLS))

n_models_saved = 0

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n── Seed {seed} (seed {seed_idx + 1}/{len(SEEDS)}) ──")

    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y.astype(str)), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        params = {**XGB_PARAMS, 'random_state': seed}

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_preds          = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        fold_rmse          = mean_squared_error(y_val, val_preds) ** 0.5

        # XGBoost importance scores keyed by feature index — align to FEATURE_COLS order
        booster    = model.get_booster()
        gain_map   = booster.get_score(importance_type='gain')
        weight_map = booster.get_score(importance_type='weight')
        for i in range(len(FEATURE_COLS)):
            importance_gain[i]   += gain_map.get(f'f{i}', 0.0)
            importance_weight[i] += weight_map.get(f'f{i}', 0.0)

        # Save model — predict_xgb.py globs xgb_s*_fold*.json
        model_path = os.path.join(MODELS_DIR, f'xgb_s{seed}_fold{fold}.json')
        model.get_booster().save_model(model_path)
        n_models_saved += 1

        print(f"  Fold {fold:2d} | best iter: {model.best_iteration:4d} | val RMSE: {fold_rmse:.4f}")

    seed_oof_rmse = mean_squared_error(y, oof_preds) ** 0.5
    all_oof_preds[seed_idx] = oof_preds
    print(f"  → Seed {seed} OOF RMSE: {seed_oof_rmse:.4f}")

# =============================================================================
# SECTION 3b – EVALUATION
# =============================================================================
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

# Average importances across all models for a stable estimate
importance_gain   /= (len(SEEDS) * N_FOLDS)
importance_weight /= (len(SEEDS) * N_FOLDS)

# =============================================================================
# SECTION 4 – OOF PREDICTION ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("SECTION 4 – OOF PREDICTION ANALYSIS")
print("="*60)

oof_df = pd.DataFrame({'true': y, 'pred': mean_oof_preds})
oof_df['error'] = oof_df['pred'] - oof_df['true']
print(oof_df['error'].describe().round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(oof_df['true'], oof_df['pred'], alpha=0.25, s=12, color='darkorange')
axes[0].plot([0.5, 6], [0.5, 6], 'r--', linewidth=1.2, label='Perfect prediction')
axes[0].set_xlabel('True Score'); axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'OOF: True vs Predicted\nEnsemble RMSE = {oof_rmse:.4f}  '
                  f'({len(SEEDS)} seeds × {N_FOLDS} folds)')
axes[0].legend()

axes[1].hist(oof_df['error'], bins=40, color='darkorange', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Prediction Error (pred − true)')
axes[1].set_ylabel('Count')
axes[1].set_title('OOF Residual Distribution')

plt.suptitle('Out-of-Fold Predictions — XGBoost', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_oof_predictions.png'), dpi=150)
plt.close()
print("\n[Saved] 01_oof_predictions.png")

# =============================================================================
# SECTION 5 – FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*60)
print("SECTION 5 – FEATURE IMPORTANCE")
print("="*60)

# Two importance types:
# - gain   : avg loss reduction each time the feature is used — reflects TRUE usefulness
# - weight : how often the feature appears as a split — can be inflated for continuous features

imp_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'gain':    importance_gain,
    'weight':  importance_weight,
}).sort_values('gain', ascending=False)

print(f"\nFeature importance (gain, averaged over {len(SEEDS) * N_FOLDS} models):")
print(imp_df.to_string(index=False))

legend_elements = [Patch(facecolor=c, label=k) for k, c in CAT_PALETTE.items()]
feat_arr        = np.array(FEATURE_COLS)

def plot_importance(imp_series, feat_names, title, filename, xlabel):
    order    = imp_series.argsort()
    sorted_f = feat_names[order]
    sorted_v = imp_series[order]
    colors   = [CAT_PALETTE[CATEGORIES[f]] for f in sorted_f]

    fig, ax = plt.subplots(figsize=(10, max(8, len(feat_names) * 0.18)))
    bars = ax.barh(sorted_f, sorted_v, color=colors, edgecolor='white')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    total = sorted_v.sum()
    for bar, v in zip(bars, sorted_v):
        ax.text(v + total * 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v / total * 100:.1f}%', va='center', ha='left', fontsize=7)
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Category')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()
    print(f"[Saved] {filename}")

# Plot 2 – Gain importance
plot_importance(
    importance_gain, feat_arr,
    title    = f'Feature Importance: Gain (avg over {len(SEEDS) * N_FOLDS} models) — XGBoost',
    filename = '02_importance_gain.png',
    xlabel   = 'Average Gain',
)

# Plot 3 – Weight importance
plot_importance(
    importance_weight, feat_arr,
    title    = f'Feature Importance: Weight (split frequency, avg over {len(SEEDS) * N_FOLDS} models) — XGBoost',
    filename = '03_importance_weight.png',
    xlabel   = 'Average Weight (split count)',
)

# Plot 4 – Gain vs Weight scatter
fig, ax = plt.subplots(figsize=(9, 7))
for feat, g, w in zip(FEATURE_COLS, importance_gain, importance_weight):
    color = CAT_PALETTE[CATEGORIES[feat]]
    ax.scatter(w, g, color=color, s=60, zorder=3)
    ax.annotate(feat, (w, g), fontsize=7, xytext=(4, 2), textcoords='offset points')
ax.set_xlabel('Weight Importance (frequency)', fontsize=11)
ax.set_ylabel('Gain Importance (loss reduction)', fontsize=11)
ax.set_title('Gain vs Weight Importance\n(discrepancies reveal feature behaviour) — XGBoost',
             fontsize=11, fontweight='bold')
ax.legend(handles=legend_elements, fontsize=9, title='Category')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '04_gain_vs_weight.png'), dpi=150)
plt.close()
print("[Saved] 04_gain_vs_weight.png")

print(f"\nAll plots saved to: {PLOT_DIR}/")
print("Done.")
