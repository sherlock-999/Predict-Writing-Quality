"""
XGBoost Regression: Predicting Essay Score from Writing Process Features
========================================================================
Pipeline:
  1. Feature engineering  — same 120 features as train_lgbm.py
  2. 5-fold stratified CV  — stratify on score bins so every fold is representative
  3. XGBoost regression    — optimised with early stopping per fold
  4. Evaluation            — CV RMSE + per-fold scores
  5. Feature importance    — gain-based and weight-based plots

Output plots: writing_process/xgb_plots/
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from important_features import compute_features, FEATURE_COLS, CATEGORIES, CAT_PALETTE

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'xgb_plots')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# SECTION 1 – FEATURE ENGINEERING
# =============================================================================
print("Loading data...")
logs   = pd.read_csv(os.path.join(DATA_DIR, 'train_logs.csv'))
scores = pd.read_csv(os.path.join(DATA_DIR, 'train_scores.csv'))

print("Computing per-essay features...")
df = compute_features(logs).merge(scores, on='id').fillna(0)
print(f"Feature matrix: {df.shape[0]} essays × {df.shape[1] - 2} features")

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
    'objective':         'reg:squarederror',
    'eval_metric':       'rmse',
    'max_depth':          6,
    'min_child_weight':   5,
    'learning_rate':      0.05,
    'subsample':          0.8,
    'colsample_bytree':   0.8,
    'reg_alpha':          0.1,
    'reg_lambda':         1.0,
    'n_estimators':       2000,
    'verbosity':          0,
    'random_state':       42,
    'device':             'cpu',
    'early_stopping_rounds': 50,
}

# =============================================================================
# SECTION 3 – 5-FOLD STRATIFIED CROSS-VALIDATION
# =============================================================================
# Stratify on score bins so every fold has a representative mix of high/low scorers.
print("\n" + "="*60)
print("SECTION 3 – 5-FOLD STRATIFIED CV")
print("="*60)

score_bins = pd.cut(y, bins=5, labels=False)
skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds   = np.zeros(len(y))
fold_rmses  = []
fold_models = []

# Accumulate feature importances (gain and weight) across folds
importance_gain   = np.zeros(len(FEATURE_COLS))
importance_weight = np.zeros(len(FEATURE_COLS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, score_bins), start=1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds          = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    fold_rmse          = mean_squared_error(y_val, val_preds) ** 0.5
    fold_rmses.append(fold_rmse)
    fold_models.append(model)

    # XGBoost importance scores keyed by feature name — align to FEATURE_COLS order
    booster = model.get_booster()
    gain_map   = booster.get_score(importance_type='gain')
    weight_map = booster.get_score(importance_type='weight')
    for i, feat in enumerate(FEATURE_COLS):
        importance_gain[i]   += gain_map.get(f'f{i}', 0.0)
        importance_weight[i] += weight_map.get(f'f{i}', 0.0)

    best_iter = model.best_iteration
    print(f"  Fold {fold} | best iter: {best_iter:4d} | val RMSE: {fold_rmse:.4f}")

# OOF RMSE — primary performance metric
oof_rmse      = mean_squared_error(y, oof_preds) ** 0.5
baseline_rmse = mean_squared_error(y, np.full_like(y, y.mean())) ** 0.5

print(f"\n  CV RMSE  (mean ± std): {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"  OOF RMSE (all folds) : {oof_rmse:.4f}")
print(f"  Baseline RMSE (mean) : {baseline_rmse:.4f}  ← predicting mean score for everyone")
print(f"  Improvement over baseline: {baseline_rmse - oof_rmse:.4f}")

# Save each fold model in XGBoost's native JSON format
for fold_idx, model in enumerate(fold_models, start=1):
    path = os.path.join(MODELS_DIR, f'xgb_fold{fold_idx}.json')
    model.get_booster().save_model(path)
print(f"\n  Saved {len(fold_models)} fold models to: {MODELS_DIR}/")

# Average importances across folds for a stable estimate
importance_gain   /= 5
importance_weight /= 5

# =============================================================================
# SECTION 4 – OOF PREDICTION ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("SECTION 4 – OOF PREDICTION ANALYSIS")
print("="*60)

oof_df = pd.DataFrame({'true': y, 'pred': oof_preds})
oof_df['error'] = oof_df['pred'] - oof_df['true']
print(oof_df['error'].describe().round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(oof_df['true'], oof_df['pred'], alpha=0.25, s=12, color='darkorange')
axes[0].plot([0.5, 6], [0.5, 6], 'r--', linewidth=1.2, label='Perfect prediction')
axes[0].set_xlabel('True Score'); axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'OOF: True vs Predicted\nRMSE = {oof_rmse:.4f}')
axes[0].legend()

axes[1].hist(oof_df['error'], bins=40, color='darkorange', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Prediction Error (pred − true)')
axes[1].set_ylabel('Count')
axes[1].set_title('OOF Residual Distribution')

plt.suptitle('Out-of-Fold Predictions — XGBoost', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_oof_predictions.png'), dpi=150)
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

print("\nFeature importance (gain, averaged over 5 folds):")
print(imp_df.to_string(index=False))

legend_elements = [Patch(facecolor=c, label=k) for k, c in CAT_PALETTE.items()]
feat_arr        = np.array(FEATURE_COLS)

def plot_importance(imp_series, feat_names, title, filename, xlabel):
    order    = imp_series.argsort()
    sorted_f = feat_names[order]
    sorted_v = imp_series[order]
    colors   = [CAT_PALETTE[CATEGORIES[f]] for f in sorted_f]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_f, sorted_v, color=colors, edgecolor='white')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    total = sorted_v.sum()
    for bar, v in zip(bars, sorted_v):
        if total > 0:
            ax.text(v + total * 0.003, bar.get_y() + bar.get_height() / 2,
                    f'{v / total * 100:.1f}%', va='center', ha='left', fontsize=8)
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Category')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"[Saved] {filename}")

# Plot 2 – Gain importance
plot_importance(
    importance_gain, feat_arr,
    title    = 'Feature Importance: Gain (avg loss reduction per split)\nAveraged over 5 CV folds — XGBoost',
    filename = '02_importance_gain.png',
    xlabel   = 'Average Gain',
)

# Plot 3 – Weight importance
plot_importance(
    importance_weight, feat_arr,
    title    = 'Feature Importance: Weight (split frequency)\nAveraged over 5 CV folds — XGBoost',
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
plt.savefig(os.path.join(OUTPUT_DIR, '04_gain_vs_weight.png'), dpi=150)
plt.close()
print("[Saved] 04_gain_vs_weight.png")

print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("Done.")
