"""
LightGBM Regression: Predicting Essay Score from Writing Process Features
=========================================================================
Pipeline:
  1. Feature engineering  — testing_new_features (loads CSV cache if available)
  2. 5-seed × 10-fold stratified CV  — 50 models total for stable OOF
  3. LightGBM regression   — optimised with early stopping per fold
  4. Evaluation            — CV RMSE + per-fold scores, OOF averaged across seeds
  5. Feature importance    — gain-based and split-based plots

Output:
  models/lgbm_s{seed}_fold{fold}.txt  — 50 trained models
  lgbm_plots/01_oof_predictions.png
  lgbm_plots/02_importance_gain.png
  lgbm_plots/03_importance_split.png

Usage:
    conda run -n exp python train_lgbm.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from v4_features import compute_features, FEATURE_COLS, CATEGORIES, CAT_PALETTE

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'v4_feature_lgbm')
PLOT_DIR    = os.path.join(BASE_DIR, 'lgbm_plots')
TFIDF_PATH  = os.path.join(MODELS_DIR, 'tfidf', 'tfidf_svd.pkl')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── CV config ─────────────────────────────────────────────────────────────────
N_FOLDS = 10
SEEDS   = [42, 21, 2022, 7, 4]   # diverse seeds for better ensemble

# ── Model parameters ──────────────────────────────────────────────────────────
# Load from Optuna output if available; fall back to reasonable defaults.
best_params_path = os.path.join(MODELS_DIR, 'lgbm_best_params.json')
if os.path.exists(best_params_path):
    with open(best_params_path) as f:
        best_params = json.load(f)
    print(f"Loaded best params from: {best_params_path}")
else:
    print("lgbm_best_params.json not found — using default params")
    best_params = {
        'num_leaves':        31,
        'min_child_samples': 20,
        'learning_rate':     0.05,
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      1,
        'lambda_l1':         0.1,
        'lambda_l2':         0.1,
    }

LGBM_PARAMS = {
    'num_leaves':        31,
        'min_child_samples': 20,
        'learning_rate':     0.05,
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      1,
        'lambda_l1':         0.1,
        'lambda_l2':         0.1,
}
'''
{
    'n_estimators':      1000,       # high ceiling — early stopping finds actual optimum
    'objective':         'regression',
    'metric':            'rmse',
    'num_leaves':        24,
    'min_child_samples': 5,
    'learning_rate':     0.005,
    'colsample_bytree':  0.41,
    'subsample':         0.98,
    'reg_alpha':         0.018,
    'reg_lambda':        0.675,
    'force_col_wise':    True,
    'verbosity':         0,
}

{
    'n_estimators': 1024,
    'learning_rate': 0.005,
    'metric': 'rmse',
    'random_state': 42,
    'force_col_wise': True,
    'verbosity': 0, 
    'num_leaves': 24, 
}
{'reg_alpha': 0.007678095440286993, 
               'reg_lambda': 0.34230534302168353, 
               'colsample_bytree': 0.627061253588415, 
               'subsample': 0.854942238828458, 
               'learning_rate': 0.038697981947473245, 
               'num_leaves': 22, 
               'max_depth': 37, 
               'min_child_samples': 18,
               'random_state': BASE_SEED ,
               'n_estimators': 150,
               "objective": "regression",
               "metric": "rmse",
               'force_col_wise': True,
               "verbosity": 0,
              }
'''



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

# Accumulate OOF predictions across seeds then average for final RMSE.
all_oof_preds   = np.zeros((len(SEEDS), len(y)))

# Accumulate feature importances across all 50 models.
importance_gain  = np.zeros(len(FEATURE_COLS))
importance_split = np.zeros(len(FEATURE_COLS))

n_models_saved = 0

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n── Seed {seed} (seed {seed_idx + 1}/{len(SEEDS)}) ──")

    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y.astype(str)), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        params = {**LGBM_PARAMS, 'random_state': seed}

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_preds          = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        fold_rmse          = mean_squared_error(y_val, val_preds) ** 0.5

        importance_gain  += model.booster_.feature_importance(importance_type='gain')
        importance_split += model.booster_.feature_importance(importance_type='split')

        # Save model — predict_lgbm.py globs lgbm_s*_fold*.txt
        model_path = os.path.join(MODELS_DIR, f'lgbm_s{seed}_fold{fold}.txt')
        model.booster_.save_model(model_path)
        n_models_saved += 1

        print(f"  Fold {fold:2d} | best iter: {model.best_iteration_:4d} | val RMSE: {fold_rmse:.4f}")

    seed_oof_rmse = mean_squared_error(y, oof_preds) ** 0.5
    all_oof_preds[seed_idx] = oof_preds
    print(f"  → Seed {seed} OOF RMSE: {seed_oof_rmse:.4f}")

# =============================================================================
# SECTION 3 – EVALUATION
# =============================================================================
print(f"\n{'='*60}")
print("SECTION 3 – EVALUATION")
print(f"{'='*60}")

# Average the OOF predictions across all seeds for the final ensemble estimate.
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

# =============================================================================
# SECTION 4 – OOF PREDICTION PLOTS
# =============================================================================
oof_df         = pd.DataFrame({'true': y, 'pred': mean_oof_preds})
oof_df['error'] = oof_df['pred'] - oof_df['true']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(oof_df['true'], oof_df['pred'], alpha=0.25, s=12, color='steelblue')
axes[0].plot([0.5, 6], [0.5, 6], 'r--', linewidth=1.2, label='Perfect prediction')
axes[0].set_xlabel('True Score'); axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'OOF: True vs Predicted\nEnsemble RMSE = {oof_rmse:.4f}  '
                  f'({len(SEEDS)} seeds × {N_FOLDS} folds)')
axes[0].legend()

axes[1].hist(oof_df['error'], bins=40, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Prediction Error (pred − true)')
axes[1].set_ylabel('Count')
axes[1].set_title('OOF Residual Distribution')

plt.suptitle('Out-of-Fold Predictions', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_oof_predictions.png'), dpi=150)
plt.close()
print("\n[Saved] 01_oof_predictions.png")

# =============================================================================
# SECTION 5 – FEATURE IMPORTANCE
# =============================================================================
# Average over all 50 models for a stable estimate.
importance_gain  /= (len(SEEDS) * N_FOLDS)
importance_split /= (len(SEEDS) * N_FOLDS)

imp_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'gain':    importance_gain,
    'split':   importance_split,
}).sort_values('gain', ascending=False)

print("\nTop 20 features by gain:")
print(imp_df.head(20).to_string(index=False))

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=k) for k, c in CAT_PALETTE.items()]


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


feat_arr = np.array(FEATURE_COLS)

plot_importance(
    importance_gain, feat_arr,
    title    = f'Feature Importance: Gain (avg over {len(SEEDS) * N_FOLDS} models)',
    filename = '02_importance_gain.png',
    xlabel   = 'Average Gain',
)

plot_importance(
    importance_split, feat_arr,
    title    = f'Feature Importance: Split Count (avg over {len(SEEDS) * N_FOLDS} models)',
    filename = '03_importance_split.png',
    xlabel   = 'Average Split Count',
)

print(f"\nAll plots saved to: {PLOT_DIR}/")
print("Done.")
