"""
OOF Correlation Analysis: LightGBM vs XGBoost vs CatBoost
==========================================================
Recomputes out-of-fold predictions for all three model families using the
saved fold models and the training feature cache, then plots correlation.

Usage:
    conda run -n exp python oof_correlation.py
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from v6_features import FEATURE_COLS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
DATA_DIR        = os.path.join(BASE_DIR, 'data')
LGBM_MODELS_DIR = os.path.join(BASE_DIR, 'v6_features_lgbm')
XGB_MODELS_DIR  = os.path.join(BASE_DIR, 'xgb_model')
CB_MODELS_DIR   = os.path.join(BASE_DIR, 'catboost_model')
PLOT_DIR        = os.path.join(BASE_DIR, 'oof_correlation_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

N_FOLDS = 10
SEEDS   = [42, 21, 2022, 7, 4]

# ── Load training data ────────────────────────────────────────────────────────
features_cache = os.path.join(DATA_DIR, 'train_features_v6.csv')

print(f"Loading features from {features_cache}...")
df = pd.read_csv(features_cache).fillna(0)

X = df[FEATURE_COLS].values
y = df['score'].values
print(f"  {len(y)} essays × {X.shape[1]} features")

# ── Recompute OOF predictions ─────────────────────────────────────────────────
# Each model family: average OOF preds across all 5 seeds
lgbm_oof = np.zeros((len(SEEDS), len(y)))
xgb_oof  = np.zeros((len(SEEDS), len(y)))
cb_oof   = np.zeros((len(SEEDS), len(y)))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n── Seed {seed} ──")
    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    lgbm_seed = np.zeros(len(y))
    xgb_seed  = np.zeros(len(y))
    cb_seed   = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y.astype(str)), start=1):
        X_val = X[val_idx]

        # LightGBM
        lgbm_path = os.path.join(LGBM_MODELS_DIR, f'lgbm_s{seed}_fold{fold}.txt')
        lgbm_model = lgb.Booster(model_file=lgbm_path)
        lgbm_seed[val_idx] = lgbm_model.predict(X_val)

        # XGBoost
        xgb_path  = os.path.join(XGB_MODELS_DIR, f'xgb_s{seed}_fold{fold}.json')
        xgb_model = xgb.Booster()
        xgb_model.load_model(xgb_path)
        dval = xgb.DMatrix(X_val, feature_names=FEATURE_COLS)
        xgb_seed[val_idx] = xgb_model.predict(dval)

        # CatBoost
        cb_path   = os.path.join(CB_MODELS_DIR, f'catboost_s{seed}_fold{fold}.cbm')
        cb_model  = CatBoostRegressor()
        cb_model.load_model(cb_path)
        cb_seed[val_idx] = cb_model.predict(X_val)

        print(f"  Fold {fold:2d} done")

    lgbm_oof[seed_idx] = lgbm_seed
    xgb_oof[seed_idx]  = xgb_seed
    cb_oof[seed_idx]   = cb_seed

# Average across seeds
lgbm_mean = lgbm_oof.mean(axis=0)
xgb_mean  = xgb_oof.mean(axis=0)
cb_mean   = cb_oof.mean(axis=0)

# ── Print RMSE ────────────────────────────────────────────────────────────────
def rmse(a, b): return mean_squared_error(a, b) ** 0.5

print("\n" + "="*50)
print("OOF RMSE (ensemble-averaged across 5 seeds)")
print("="*50)
print(f"  LGBM     : {rmse(y, lgbm_mean):.6f}")
print(f"  XGB      : {rmse(y, xgb_mean):.6f}")
print(f"  CatBoost : {rmse(y, cb_mean):.6f}")

ab   = (lgbm_mean + xgb_mean) / 2
ac   = (lgbm_mean + cb_mean)  / 2
bc   = (xgb_mean  + cb_mean)  / 2
abc  = (lgbm_mean + xgb_mean + cb_mean) / 3

print(f"\n  A+B      : {rmse(y, ab):.6f}")
print(f"  A+C      : {rmse(y, ac):.6f}")
print(f"  B+C      : {rmse(y, bc):.6f}")
print(f"  A+B+C    : {rmse(y, abc):.6f}")

# ── Correlation matrix ────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Pearson correlation between OOF predictions")
print("="*50)
labels = ['LGBM', 'XGB', 'CatBoost']
preds  = np.stack([lgbm_mean, xgb_mean, cb_mean], axis=1)
corr   = np.corrcoef(preds.T)

for i, li in enumerate(labels):
    for j, lj in enumerate(labels):
        if j > i:
            print(f"  {li} vs {lj}: {corr[i, j]:.6f}")

# ── PLOT 1: Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr, vmin=0.95, vmax=1.0, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=11)
ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=11)
ax.set_title('OOF Prediction Correlation', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{corr[i,j]:.4f}', ha='center', va='center',
                fontsize=11, color='black' if corr[i,j] < 0.985 else 'white')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_correlation_heatmap.png'), dpi=150)
plt.close()
print("\n[Saved] 01_correlation_heatmap.png")

# ── PLOT 2: Scatter matrix of pairwise OOF predictions ───────────────────────
pairs   = [('LGBM', lgbm_mean, 'XGB', xgb_mean),
           ('LGBM', lgbm_mean, 'CatBoost', cb_mean),
           ('XGB',  xgb_mean,  'CatBoost', cb_mean)]
colors  = ['#2196F3', '#FF9800', '#9C27B0']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (lx, px, ly, py), c in zip(axes, pairs, colors):
    ax.scatter(px, py, alpha=0.15, s=8, color=c)
    lims = [min(px.min(), py.min()) - 0.1, max(px.max(), py.max()) + 0.1]
    ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
    r = np.corrcoef(px, py)[0, 1]
    ax.set_xlabel(f'{lx} OOF prediction', fontsize=10)
    ax.set_ylabel(f'{ly} OOF prediction', fontsize=10)
    ax.set_title(f'{lx} vs {ly}\nr = {r:.5f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle('Pairwise OOF Prediction Scatter', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_scatter_matrix.png'), dpi=150)
plt.close()
print("[Saved] 02_scatter_matrix.png")

# ── PLOT 3: Residual comparison — where do models disagree? ──────────────────
res_lgbm = lgbm_mean - y
res_xgb  = xgb_mean  - y
res_cb   = cb_mean   - y

# Disagreement = std of the three predictions per sample
disagreement = np.stack([lgbm_mean, xgb_mean, cb_mean], axis=0).std(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Residuals overlay
bins = np.linspace(-2, 2, 60)
axes[0].hist(res_lgbm, bins=bins, alpha=0.5, label='LGBM',     color='#2196F3', edgecolor='none')
axes[0].hist(res_xgb,  bins=bins, alpha=0.5, label='XGB',      color='#FF9800', edgecolor='none')
axes[0].hist(res_cb,   bins=bins, alpha=0.5, label='CatBoost', color='#9C27B0', edgecolor='none')
axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('OOF Residual (pred − true)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('OOF Residual Distributions', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)

# Disagreement vs true score
axes[1].scatter(y, disagreement, alpha=0.2, s=10, color='steelblue')
axes[1].set_xlabel('True Score', fontsize=11)
axes[1].set_ylabel('Std of 3 OOF predictions (disagreement)', fontsize=11)
axes[1].set_title('Where Do Models Disagree?', fontsize=12, fontweight='bold')

# Annotate mean disagreement per score bin
for score_val in sorted(np.unique(y)):
    mask = y == score_val
    mean_d = disagreement[mask].mean()
    axes[1].plot(score_val, mean_d, 'ro', markersize=8, zorder=5)
    axes[1].annotate(f'{mean_d:.3f}', (score_val, mean_d),
                     xytext=(4, 4), textcoords='offset points', fontsize=8, color='red')

plt.suptitle('OOF Residuals & Model Disagreement', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '03_residuals_and_disagreement.png'), dpi=150)
plt.close()
print("[Saved] 03_residuals_and_disagreement.png")

# ── PLOT 4: Optimal blend weight search (LGBM vs XGB, best pair) ─────────────
weights = np.linspace(0, 1, 201)
blend_rmses = [rmse(y, w * lgbm_mean + (1-w) * xgb_mean) for w in weights]
best_w = weights[np.argmin(blend_rmses)]
best_r = min(blend_rmses)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(weights, blend_rmses, color='#2196F3', linewidth=2)
ax.axvline(best_w, color='red', linestyle='--', linewidth=1.5,
           label=f'Best w={best_w:.2f}  RMSE={best_r:.6f}')
ax.axhline(rmse(y, lgbm_mean), color='steelblue', linestyle=':', linewidth=1, label=f'LGBM alone: {rmse(y, lgbm_mean):.6f}')
ax.axhline(rmse(y, xgb_mean),  color='darkorange', linestyle=':', linewidth=1, label=f'XGB alone:  {rmse(y, xgb_mean):.6f}')
ax.set_xlabel('Weight on LGBM  (1−w on XGB)', fontsize=11)
ax.set_ylabel('OOF RMSE', fontsize=11)
ax.set_title('Optimal Blend Weight: LGBM vs XGB', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '04_blend_weight_search.png'), dpi=150)
plt.close()
print(f"[Saved] 04_blend_weight_search.png  (best w_lgbm={best_w:.2f}, RMSE={best_r:.6f})")

print(f"\nAll plots saved to: {PLOT_DIR}/")
print("Done.")
