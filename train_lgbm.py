"""
LightGBM Regression: Predicting Essay Score from Writing Process Features
=========================================================================
Pipeline:
  1. Feature engineering  — same 19 features as eda.py
  2. 5-fold stratified CV  — stratify on score bins so every fold is representative
  3. LightGBM regression   — optimised with early stopping per fold
  4. Evaluation            — CV RMSE + per-fold scores
  5. Feature importance    — gain-based and split-based plots

Output plots: writing_process/lgbm_plots/
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'lgbm_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1 – FEATURE ENGINEERING  (mirrors eda.py)
# =============================================================================
print("Loading data...")
logs   = pd.read_csv(os.path.join(DATA_DIR, 'train_logs.csv'))
scores = pd.read_csv(os.path.join(DATA_DIR, 'train_scores.csv'))
logs   = logs.sort_values(['id', 'event_id']).reset_index(drop=True)
logs['iki'] = logs.groupby('id')['down_time'].diff()


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


rows = []
print("Computing per-essay features...")
for essay_id, essay in logs.groupby('id'):
    essay    = essay.reset_index(drop=True)
    n        = len(essay)
    gaps     = essay.loc[essay['activity'] == 'Input', 'iki'].dropna()
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

    total_time_ms  = essay['down_time'].max() - essay['down_time'].min()
    active_time_ms = max(total_time_ms - all_gaps[all_gaps > 2_000].sum(), 1)
    typing_wpm     = final_word_count / (active_time_ms / 60_000)

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
        'final_word_count':      final_word_count,
        'total_events':          total_events,
        'median_iki_ms':         median_iki_ms,
        'iki_std':               iki_std,
        'typing_wpm':            typing_wpm,
        'pauses_over_2s':        pauses_over_2s,
        'pauses_over_5s':        pauses_over_5s,
        'mean_long_pause_ms':    mean_long_pause_ms,
        'deletion_ratio':        deletion_ratio,
        'global_revision_count': global_revision_count,
        'local_revision_count':  local_revision_count,
        'period_count':          period_count,
        'comma_count':           comma_count,
        'semicolon_count':       semicolon_count,
        'newline_count':         newline_count,
        'wpm_early':             wpm_early,
        'wpm_middle':            wpm_middle,
        'wpm_late':              wpm_late,
        'wpm_acceleration':      wpm_acceleration,
    })

df = pd.DataFrame(rows).merge(scores, on='id').fillna(0)
print(f"Feature matrix: {df.shape[0]} essays × {df.shape[1] - 2} features")

FEATURE_COLS = [c for c in df.columns if c not in ['id', 'score']]
X = df[FEATURE_COLS].values
y = df['score'].values

# =============================================================================
# SECTION 2 – MODEL CONFIGURATION
# =============================================================================

# LightGBM parameters.
# - objective='regression'    : minimise MSE (equivalent to predicting mean)
# - metric='rmse'             : report root-mean-squared-error during training
# - num_leaves=31             : max leaves per tree; lower = more regularised
# - min_child_samples=20      : min samples in a leaf; prevents fitting to tiny groups
# - learning_rate=0.05        : small steps so early stopping can find the right depth
# - feature_fraction=0.8      : each tree sees 80% of features — reduces correlation between trees
# - bagging_fraction/freq     : row-level subsampling — another regulariser
# - lambda_l1/l2              : explicit L1/L2 weight penalty on leaf values
# - n_estimators=2000         : upper bound; early stopping will find the true optimum
# - early_stopping_rounds=50  : stop if val RMSE hasn't improved for 50 rounds
LGBM_PARAMS = {
    'objective':        'regression',
    'metric':           'rmse',
    'num_leaves':        31,
    'min_child_samples': 20,
    'learning_rate':     0.05,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      1,
    'lambda_l1':         0.1,
    'lambda_l2':         0.1,
    'n_estimators':      2000,
    'verbose':          -1,
    'random_state':      42,
}

# =============================================================================
# SECTION 3 – 5-FOLD STRATIFIED CROSS-VALIDATION
# =============================================================================
# Stratify on score bins so every fold has a representative mix of high/low scorers.
# With only 2,471 samples a pure random split could produce unbalanced folds.
print("\n" + "="*60)
print("SECTION 3 – 5-FOLD STRATIFIED CV")
print("="*60)

score_bins   = pd.cut(y, bins=5, labels=False)   # 5 equal-width bins for stratification
skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds    = np.zeros(len(y))     # out-of-fold predictions collected across all folds
fold_rmses   = []
fold_models  = []

# Accumulate feature importances across folds and average them at the end
importance_gain  = np.zeros(len(FEATURE_COLS))
importance_split = np.zeros(len(FEATURE_COLS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, score_bins), start=1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),   # suppress per-round output
        ],
    )

    val_preds           = model.predict(X_val)
    oof_preds[val_idx]  = val_preds
    fold_rmse           = mean_squared_error(y_val, val_preds) ** 0.5
    fold_rmses.append(fold_rmse)
    fold_models.append(model)

    importance_gain  += model.booster_.feature_importance(importance_type='gain')
    importance_split += model.booster_.feature_importance(importance_type='split')

    best_iter = model.best_iteration_
    print(f"  Fold {fold} | best iter: {best_iter:4d} | val RMSE: {fold_rmse:.4f}")

# Overall OOF RMSE — this is the single number that represents model performance
oof_rmse = mean_squared_error(y, oof_preds) ** 0.5
baseline_rmse = mean_squared_error(y, np.full_like(y, y.mean())) ** 0.5

print(f"\n  CV RMSE  (mean ± std): {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"  OOF RMSE (all folds) : {oof_rmse:.4f}")
print(f"  Baseline RMSE (mean) : {baseline_rmse:.4f}  ← predicting mean score for everyone")
print(f"  Improvement over baseline: {baseline_rmse - oof_rmse:.4f}")

# Save each fold model so predict.py can load them without retraining.
# LightGBM's native .txt format is portable and version-stable.
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
for fold_idx, model in enumerate(fold_models, start=1):
    path = os.path.join(MODELS_DIR, f'lgbm_fold{fold_idx}.txt')
    model.booster_.save_model(path)
print(f"\n  Saved {len(fold_models)} fold models to: {MODELS_DIR}/")

# Average importances across folds for a stable estimate
importance_gain  /= 5
importance_split /= 5

# =============================================================================
# SECTION 4 – OOF PREDICTION ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("SECTION 4 – OOF PREDICTION ANALYSIS")
print("="*60)

oof_df = pd.DataFrame({'true': y, 'pred': oof_preds})
oof_df['error'] = oof_df['pred'] - oof_df['true']
print(oof_df['error'].describe().round(4))

# Plot 1 – True vs predicted scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(oof_df['true'], oof_df['pred'], alpha=0.25, s=12, color='steelblue')
axes[0].plot([0.5, 6], [0.5, 6], 'r--', linewidth=1.2, label='Perfect prediction')
axes[0].set_xlabel('True Score'); axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'OOF: True vs Predicted\nRMSE = {oof_rmse:.4f}')
axes[0].legend()

axes[1].hist(oof_df['error'], bins=40, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Prediction Error (pred − true)')
axes[1].set_ylabel('Count')
axes[1].set_title('OOF Residual Distribution')

plt.suptitle('Out-of-Fold Predictions', fontweight='bold')
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
# - gain  : avg reduction in loss each time the feature is used — reflects TRUE usefulness
# - split : how often the feature appears as a split — can be inflated for low-cardinality features

imp_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'gain':    importance_gain,
    'split':   importance_split,
}).sort_values('gain', ascending=False)

print("\nFeature importance (gain, averaged over 5 folds):")
print(imp_df.to_string(index=False))

# Category colours — same palette as eda.py
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
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=k) for k, c in cat_palette.items()]

def plot_importance(imp_series, feat_names, title, filename, xlabel):
    order      = imp_series.argsort()
    sorted_f   = feat_names[order]
    sorted_v   = imp_series[order]
    colors     = [cat_palette[categories[f]] for f in sorted_f]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_f, sorted_v, color=colors, edgecolor='white')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    # Normalised value labels (% of total)
    total = sorted_v.sum()
    for bar, v in zip(bars, sorted_v):
        ax.text(v + total * 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v / total * 100:.1f}%', va='center', ha='left', fontsize=8)
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Category')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"[Saved] {filename}")

feat_arr = np.array(FEATURE_COLS)

# Plot 2 – Gain importance (primary — reflects actual loss reduction)
plot_importance(
    importance_gain, feat_arr,
    title    = 'Feature Importance: Gain (avg loss reduction per split)\nAveraged over 5 CV folds',
    filename = '02_importance_gain.png',
    xlabel   = 'Average Gain',
)

# Plot 3 – Split importance (secondary — how often each feature is chosen as a split)
plot_importance(
    importance_split, feat_arr,
    title    = 'Feature Importance: Split Count (how often used as a split)\nAveraged over 5 CV folds',
    filename = '03_importance_split.png',
    xlabel   = 'Average Split Count',
)

# Plot 4 – Gain vs Split scatter to spot discrepancies
# A feature high in gain but low in split = a few very important cuts
# A feature high in split but low in gain = used often but not very decisive
fig, ax = plt.subplots(figsize=(9, 7))
for feat, g, s in zip(FEATURE_COLS, importance_gain, importance_split):
    color = cat_palette[categories[feat]]
    ax.scatter(s, g, color=color, s=60, zorder=3)
    ax.annotate(feat, (s, g), fontsize=7, xytext=(4, 2), textcoords='offset points')
ax.set_xlabel('Split Importance (frequency)', fontsize=11)
ax.set_ylabel('Gain Importance (loss reduction)', fontsize=11)
ax.set_title('Gain vs Split Importance\n(discrepancies reveal feature behaviour)', fontsize=11, fontweight='bold')
ax.legend(handles=legend_elements, fontsize=9, title='Category')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_gain_vs_split.png'), dpi=150)
plt.close()
print("[Saved] 04_gain_vs_split.png")

print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("Done.")
