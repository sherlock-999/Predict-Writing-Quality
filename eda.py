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

# Pre-compute inter-keystroke interval (IKI) — time between consecutive key-downs
# within each essay (first event of each essay gets NaN)
logs['iki'] = logs.groupby('id')['down_time'].diff()

rows = []
print("Computing per-essay features...")

def wpm_in_window(essay, t_start, t_end):
    """Words per minute added in the time window [t_start, t_end] (ms)."""
    mask    = (essay['down_time'] >= t_start) & (essay['down_time'] <= t_end)
    segment = essay[mask]
    if len(segment) == 0:
        return 0.0
    wc_start = essay.loc[essay['down_time'] < t_start, 'word_count'].max()
    wc_start = wc_start if pd.notna(wc_start) else 0
    words    = max(segment['word_count'].max() - wc_start, 0)
    minutes  = (t_end - t_start) / 60_000
    return (words / minutes) if minutes > 0 else 0.0


for essay_id, essay in logs.groupby('id'):
    essay = essay.reset_index(drop=True)
    n     = len(essay)
    gaps  = essay.loc[essay['activity'] == 'Input', 'iki'].dropna()
    all_gaps = essay['iki'].dropna()

    # ── VOLUME ────────────────────────────────────────────────────────────────
    # final_word_count : max word_count in session — length of finished essay
    # total_events     : total log rows — overall amount of writing activity
    final_word_count = essay['word_count'].max()
    total_events     = n

    # ── SPEED & FLUENCY ───────────────────────────────────────────────────────
    # median_iki_ms : median gap (ms) between Input keystrokes; lower = more fluent
    # iki_std       : std of IKI; higher = more uneven, hesitant rhythm
    # typing_wpm    : words per active minute (session time minus pauses >2 s)
    median_iki_ms = gaps.median() if len(gaps) > 0 else np.nan
    iki_std       = gaps.std()    if len(gaps) > 0 else np.nan

    total_time_ms   = essay['down_time'].max() - essay['down_time'].min()
    active_time_ms  = max(total_time_ms - all_gaps[all_gaps > 2_000].sum(), 1)
    typing_wpm      = final_word_count / (active_time_ms / 60_000)

    # ── PAUSING ───────────────────────────────────────────────────────────────
    # pauses_over_2s     : number of gaps >2 s — deliberate thinking breaks
    # pauses_over_5s     : number of gaps >5 s — longer planning/distraction episodes
    # mean_long_pause_ms : mean duration of pauses >2 s; longer = deeper pauses
    pauses_over_2s     = (all_gaps > 2_000).sum()
    pauses_over_5s     = (all_gaps > 5_000).sum()
    long_pauses        = all_gaps[all_gaps > 2_000]
    mean_long_pause_ms = long_pauses.mean() if len(long_pauses) > 0 else 0.0

    # ── REVISION ──────────────────────────────────────────────────────────────
    # deletion_ratio        : Remove/Cut events ÷ total events — fraction of time spent deleting
    # global_revision_count : events where cursor is >50 chars behind the furthest point reached;
    #                         signals jumping back to rewrite a sentence or more
    # local_revision_count  : events where cursor is ≤10 chars behind frontier — nearby typo fixes
    frontier              = essay['cursor_position'].cummax()
    dist_from_frontier    = frontier - essay['cursor_position']
    deletion_ratio        = (essay['activity'] == 'Remove/Cut').sum() / n
    global_revision_count = (dist_from_frontier > 50).sum()
    local_revision_count  = (dist_from_frontier <= 10).sum()

    # ── PUNCTUATION ───────────────────────────────────────────────────────────
    # Letters are anonymised as 'q' but punctuation in text_change is preserved.
    # period_count    : number of '.' typed — proxy for sentence count
    # comma_count     : number of ',' typed — proxy for syntactic complexity
    # semicolon_count : number of ';' typed — marker of advanced sentence construction
    # newline_count   : number of '\n' typed — proxy for paragraph count
    tc              = essay['text_change']
    period_count    = (tc == '.').sum()
    comma_count     = (tc == ',').sum()
    semicolon_count = (tc == ';').sum()
    newline_count   = (tc == '\n').sum()

    # ── WRITING MOMENTUM ──────────────────────────────────────────────────────
    # Session split into three equal time-thirds to capture production dynamics.
    # wpm_early        : words per minute in first third — how fast the writer starts
    # wpm_middle       : words per minute in middle third — often peak production
    # wpm_late         : words per minute in final third — generation vs editing trade-off
    # wpm_acceleration : wpm_middle − wpm_early; positive = writer ramped up speed
    t_min   = essay['down_time'].min()
    t_max   = essay['down_time'].max()
    t_range = max(t_max - t_min, 1)
    t1, t2  = t_min + t_range / 3, t_min + 2 * t_range / 3

    wpm_early        = wpm_in_window(essay, t_min, t1)
    wpm_middle       = wpm_in_window(essay, t1,   t2)
    wpm_late         = wpm_in_window(essay, t2,   t_max)
    wpm_acceleration = wpm_middle - wpm_early

    rows.append({
        'id': essay_id,
        # Volume
        'final_word_count':      final_word_count,
        'total_events':          total_events,
        # Speed & fluency
        'median_iki_ms':         median_iki_ms,
        'iki_std':               iki_std,
        'typing_wpm':            typing_wpm,
        # Pausing
        'pauses_over_2s':        pauses_over_2s,
        'pauses_over_5s':        pauses_over_5s,
        'mean_long_pause_ms':    mean_long_pause_ms,
        # Revision
        'deletion_ratio':        deletion_ratio,
        'global_revision_count': global_revision_count,
        'local_revision_count':  local_revision_count,
        # Punctuation
        'period_count':          period_count,
        'comma_count':           comma_count,
        'semicolon_count':       semicolon_count,
        'newline_count':         newline_count,
        # Writing momentum
        'wpm_early':             wpm_early,
        'wpm_middle':            wpm_middle,
        'wpm_late':              wpm_late,
        'wpm_acceleration':      wpm_acceleration,
    })

features = pd.DataFrame(rows).merge(scores, on='id')
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

# Category labels for colour-coding the bar chart
categories = {
    'final_word_count':      'Volume',
    'total_events':          'Volume',
    'median_iki_ms':         'Speed & Fluency',
    'iki_std':               'Speed & Fluency',
    'typing_wpm':            'Speed & Fluency',
    'pauses_over_2s':        'Pausing',
    'pauses_over_5s':        'Pausing',
    'mean_long_pause_ms':    'Pausing',
    'deletion_ratio':        'Revision',
    'global_revision_count': 'Revision',
    'local_revision_count':  'Revision',
    'period_count':          'Punctuation',
    'comma_count':           'Punctuation',
    'semicolon_count':       'Punctuation',
    'newline_count':         'Punctuation',
    'wpm_early':             'Momentum',
    'wpm_middle':            'Momentum',
    'wpm_late':              'Momentum',
    'wpm_acceleration':      'Momentum',
}
cat_palette = {
    'Volume':          '#4C72B0',
    'Speed & Fluency': '#DD8452',
    'Pausing':         '#55A868',
    'Revision':        '#C44E52',
    'Punctuation':     '#8172B2',
    'Momentum':        '#937860',
}

# Plot 2 – Ranked bar chart coloured by category
corr_sorted = corr.sort_values(ascending=True)   # ascending so highest is at top
bar_colors  = [cat_palette[categories[f]] for f in corr_sorted.index]

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
legend_elements = [Patch(facecolor=c, label=k) for k, c in cat_palette.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Category')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_feature_correlation_signed.png'), dpi=150)
plt.close()
print("\n[Saved] 02_feature_correlation_signed.png")

# Plot 2b – |r| ranked: feature importance by predictive strength regardless of direction.
# Use abs(r) when ranking importance because a feature with r = -0.57 is equally
# informative as one with r = +0.57 — the sign tells direction, not usefulness.
abs_corr        = corr.abs().sort_values(ascending=True)   # ascending so strongest is at top
abs_bar_colors  = [cat_palette[categories[f]] for f in abs_corr.index]

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
    ax.scatter(x, y, alpha=0.2, s=8, color=cat_palette[categories[feat]])
    m, b = np.polyfit(x.fillna(x.median()), y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m * xs + b, color='red', linewidth=1.5)
    r = corr[feat]
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title(f'{feat}\nr = {r:.3f}  |  [{categories[feat]}]', fontsize=9)
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
    ax.set_title(f'{feat}\n[{categories[feat]}]', fontsize=9)
    ax.set_xlabel('Score Bucket'); ax.set_ylabel(feat)
    plt.sca(ax); plt.title(f'{feat}  [{categories[feat]}]', fontsize=9)
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
