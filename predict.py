"""
Inference: Predict Essay Scores from test_logs.csv
====================================================
Loads the 5 fold models saved by train_lgbm.py, applies identical feature
engineering to test_logs.csv, averages predictions across folds, and writes
a submission.csv matching the format of sample_submission.csv.

Usage:
    conda run -n exp python predict.py

Output:
    writing_process/submission.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUT_PATH = os.path.join(BASE_DIR, 'submission.csv')

# Feature columns must match train_lgbm.py exactly — order matters for LightGBM
FEATURE_COLS = [
    'final_word_count', 'total_events',
    'median_iki_ms', 'iki_std', 'typing_wpm',
    'pauses_over_2s', 'pauses_over_5s', 'mean_long_pause_ms',
    'deletion_ratio', 'global_revision_count', 'local_revision_count',
    'period_count', 'comma_count', 'semicolon_count', 'newline_count',
    'wpm_early', 'wpm_middle', 'wpm_late', 'wpm_acceleration',
]

# =============================================================================
# SECTION 1 – LOAD MODELS
# =============================================================================
model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, 'lgbm_fold*.txt')))
if not model_paths:
    raise FileNotFoundError(
        f"No fold models found in {MODELS_DIR}/\n"
        "Run train_lgbm.py first to train and save the models."
    )

models = [lgb.Booster(model_file=p) for p in model_paths]
print(f"Loaded {len(models)} fold models from {MODELS_DIR}/")

# =============================================================================
# SECTION 2 – FEATURE ENGINEERING  (identical to train_lgbm.py)
# =============================================================================
print("Loading test_logs.csv...")
logs = pd.read_csv(os.path.join(DATA_DIR, 'test_logs.csv'))
logs = logs.sort_values(['id', 'event_id']).reset_index(drop=True)
logs['iki'] = logs.groupby('id')['down_time'].diff()
print(f"  {logs.shape[0]:,} rows | {logs['id'].nunique()} essays")


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
        'id':                    essay_id,
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

test_df = pd.DataFrame(rows).fillna(0)
X_test  = test_df[FEATURE_COLS].values
print(f"  Feature matrix: {X_test.shape[0]} essays × {X_test.shape[1]} features")

# =============================================================================
# SECTION 3 – INFERENCE
# =============================================================================
# Average predictions from all 5 fold models (ensemble).
# Each model was trained on 80% of the data; averaging reduces variance.
print("\nRunning inference...")
fold_preds = np.stack([model.predict(X_test) for model in models], axis=0)
preds      = fold_preds.mean(axis=0)

# Clip to the valid score range seen during training [0.5, 6.0].
# LightGBM can occasionally extrapolate slightly outside the training range.
preds = np.clip(preds, 0.5, 6.0)

print(f"  Prediction stats — mean: {preds.mean():.3f} | "
      f"std: {preds.std():.3f} | min: {preds.min():.3f} | max: {preds.max():.3f}")

# =============================================================================
# SECTION 4 – WRITE SUBMISSION
# =============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'score': preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to: {OUTPUT_PATH}")
print(f"  {len(submission)} rows")
print(submission.head(10).to_string(index=False))
