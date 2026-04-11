"""
EDA: Linking Writing Process to Writing Quality
================================================
Computes per-essay features from keystroke logs, correlates each with the
holistic essay score, and saves ranked correlation / scatter plots.

Output plots: writing_process/eda_plots/

NOTE: All alphabetic characters in `text_change` are anonymised as 'q'.
      Spaces, punctuation (. , ; ? ! :) and newlines are preserved,
      so punctuation-based features are fully reliable.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from features import compute_features, CATEGORIES, CAT_PALETTE

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
logs   = pd.read_csv(os.path.join(DATA_DIR, 'train_logs.csv'))
scores = pd.read_csv(os.path.join(DATA_DIR, 'train_scores.csv'))
logs   = logs.sort_values(['id', 'event_id']).reset_index(drop=True)
print(f"  train_logs  : {logs.shape[0]:,} rows × {logs.shape[1]} cols")
print(f"  train_scores: {scores.shape[0]:,} rows × {scores.shape[1]} cols")
print(f"  Unique essays: {logs['id'].nunique()}")
print(f"  Missing values — logs: {logs.isnull().sum().sum()}, scores: {scores.isnull().sum().sum()}")

# =============================================================================
# SECTION 1 – RAW DATA OVERVIEW
# =============================================================================
print("\n" + "="*60)
print("SECTION 1 – RAW DATA OVERVIEW")
print("="*60)

print("\nScore distribution (half-point scale 0.5–6.0):")
print(scores['score'].describe())
print(scores['score'].value_counts().sort_index())

print("\nActivity type breakdown:")
print(logs['activity'].value_counts())

# Plot 1a – Score distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
scores['score'].value_counts().sort_index().plot(
    kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Score Value Counts')
axes[0].set_xlabel('Score'); axes[0].set_ylabel('Count')

scores['score'].plot(kind='hist', bins=24, ax=axes[1], color='steelblue', edgecolor='white')
axes[1].axvline(scores['score'].mean(), color='red', linestyle='--',
                label=f"Mean={scores['score'].mean():.2f}")
axes[1].set_title('Score Histogram'); axes[1].set_xlabel('Score')
axes[1].legend()
plt.suptitle('Target Variable: Essay Score', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_score_distribution.png'), dpi=150)
plt.close()

# =============================================================================
# SECTION 2 – PER-ESSAY FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*60)
print("SECTION 2 – PER-ESSAY FEATURE ENGINEERING")
print("="*60)

print("Computing per-essay features...")
features = compute_features(logs).merge(scores, on='id')
print(f"Feature matrix: {features.shape[0]} essays × {features.shape[1] - 2} features")

# =============================================================================
# SECTION 3 – BASIC STATS PER FEATURE
# =============================================================================
print("\n" + "="*60)
print("SECTION 3 – FEATURE DESCRIPTIVE STATS")
print("="*60)
feat_cols = [c for c in features.columns if c not in ['id', 'score']]
print(features[feat_cols].describe().round(2).to_string())

# =============================================================================
# SECTION 4 – CORRELATION WITH SCORE
# =============================================================================
print("\n" + "="*60)
print("SECTION 4 – FEATURE CORRELATION WITH SCORE")
print("="*60)

corr = (features[feat_cols + ['score']]
        .corr()['score']
        .drop('score')
        .sort_values(ascending=False))

print("\nAll feature correlations with score (ranked):")
print(corr.to_string())

# Plot 2 – Ranked bar chart coloured by category
corr_sorted = corr.sort_values(ascending=True)   # ascending so highest is at top
bar_colors  = [CAT_PALETTE[CATEGORIES[f]] for f in corr_sorted.index]

fig, ax = plt.subplots(figsize=(10, 9))
bars = ax.barh(corr_sorted.index, corr_sorted.values, color=bar_colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Pearson r with Essay Score', fontsize=11)
ax.set_title('Signed Pearson r with Score (shows direction)\nColoured by Category', fontsize=12, fontweight='bold')
# Value labels
for bar, v in zip(bars, corr_sorted.values):
    ax.text(v + (0.004 if v >= 0 else -0.004), bar.get_y() + bar.get_height() / 2,
            f'{v:.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=k) for k, c in CAT_PALETTE.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Category')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_feature_correlation_signed.png'), dpi=150)
plt.close()
print("\n[Saved] 02_feature_correlation_signed.png")

# Plot 2b – |r| ranked: feature importance by predictive strength regardless of direction.
# Use abs(r) when ranking importance because a feature with r = -0.57 is equally
# informative as one with r = +0.57 — the sign tells direction, not usefulness.
abs_corr        = corr.abs().sort_values(ascending=True)   # ascending so strongest is at top
abs_bar_colors  = [CAT_PALETTE[CATEGORIES[f]] for f in abs_corr.index]

fig, ax = plt.subplots(figsize=(10, 9))
bars = ax.barh(abs_corr.index, abs_corr.values, color=abs_bar_colors, edgecolor='white')
ax.set_xlabel('|Pearson r| with Essay Score', fontsize=11)
ax.set_title('Feature Importance by |Pearson r| (strongest → weakest)\nColoured by Category',
             fontsize=12, fontweight='bold')
for bar, (feat, v) in zip(bars, abs_corr.items()):
    sign = '+' if corr[feat] >= 0 else '−'
    ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{sign}{v:.3f}', va='center', ha='left', fontsize=8)
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Category')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02b_feature_importance_abs_corr.png'), dpi=150)
plt.close()
print("[Saved] 02b_feature_importance_abs_corr.png")

# =============================================================================
# SECTION 5 – SCORE × TOP FEATURES SCATTER
# =============================================================================
top_features = corr.abs().sort_values(ascending=False).head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for ax, feat in zip(axes, top_features):
    x = features[feat]
    y = features['score']
    ax.scatter(x, y, alpha=0.2, s=8, color=CAT_PALETTE[CATEGORIES[feat]])
    m, b = np.polyfit(x.fillna(x.median()), y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m * xs + b, color='red', linewidth=1.5)
    r = corr[feat]
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title(f'{feat}\nr = {r:.3f}  |  [{CATEGORIES[feat]}]', fontsize=9)
plt.suptitle('Top 6 Features vs Essay Score', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_scatter_top6_features.png'), dpi=150)
plt.close()
print("[Saved] 03_scatter_top6_features.png")

# =============================================================================
# SECTION 6 – BOX PLOTS BY SCORE GROUP
# =============================================================================
features['score_bucket'] = pd.cut(
    features['score'], bins=[0, 2, 3, 4, 5, 6],
    labels=['0–2', '2–3', '3–4', '4–5', '5–6'])

plot_feats = ['final_word_count', 'typing_wpm', 'median_iki_ms',
              'period_count', 'global_revision_count', 'wpm_acceleration']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for ax, feat in zip(axes, plot_feats):
    features.boxplot(column=feat, by='score_bucket', ax=ax)
    ax.set_title(f'{feat}\n[{CATEGORIES[feat]}]', fontsize=9)
    ax.set_xlabel('Score Bucket'); ax.set_ylabel(feat)
    plt.sca(ax); plt.title(f'{feat}  [{CATEGORIES[feat]}]', fontsize=9)
plt.suptitle('Feature Distributions by Score Bucket', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_boxplots_by_score_bucket.png'), dpi=150)
plt.close()
print("[Saved] 04_boxplots_by_score_bucket.png")

# =============================================================================
# SECTION 7 – CORRELATION HEATMAP (all features)
# =============================================================================
heat_df = features[feat_cols + ['score']].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(heat_df, dtype=bool), k=1)
sns.heatmap(heat_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask, ax=ax, linewidths=0.4, annot_kws={'size': 7})
ax.set_title('Full Feature Correlation Heatmap', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_correlation_heatmap.png'), dpi=150)
plt.close()
print("[Saved] 05_correlation_heatmap.png")

# =============================================================================
# SECTION 8 – MEAN STATS BY SCORE (diagnostic table)
# =============================================================================
print("\n" + "="*60)
print("SECTION 8 – MEAN FEATURE VALUES BY SCORE")
print("="*60)
print(features.groupby('score')[feat_cols].mean().round(2).to_string())

print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("Done.")
