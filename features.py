"""
Feature Engineering: Linking Writing Process to Writing Quality
================================================================
Centralized feature computation module used by eda.py, train_lgbm.py, and predict.py.

To add a new feature:
1. Add the feature computation in compute_features()
2. Add the feature name to FEATURE_COLS
3. Add the feature to CATEGORIES for color-coding in plots

Features are organized into 5 categories:
- Volume: Essay length and activity amount
- Speed & Fluency: Typing speed and rhythm
- Pausing: Thinking/planning breaks
- Revision: Editing behavior
- Punctuation: Sentence/paragraph structure
- Momentum: Writing pace over time
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    'final_word_count', 'total_events',
    'median_iki_ms', 'iki_std', 'typing_wpm',
    'pauses_over_2s', 'pauses_over_5s', 'mean_long_pause_ms',
    'deletion_ratio', 'global_revision_count', 'local_revision_count',
    'period_count', 'comma_count', 'semicolon_count', 'newline_count',
    'wpm_early', 'wpm_middle', 'wpm_late', 'wpm_acceleration',
]

CATEGORIES = {
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

CAT_PALETTE = {
    'Volume':          '#4C72B0',
    'Speed & Fluency': '#DD8452',
    'Pausing':         '#55A868',
    'Revision':        '#C44E52',
    'Punctuation':     '#8172B2',
    'Momentum':        '#937860',
}


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


def compute_features(logs):
    """
    Compute per-essay features from keystroke logs.

    Parameters
    ----------
    logs : pd.DataFrame
        Raw keystroke logs with columns: id, event_id, down_time, up_time,
        activity, text_change, cursor_position, word_count.

    Returns
    -------
    pd.DataFrame
        One row per essay with computed features.
    """
    logs = logs.sort_values(['id', 'event_id']).reset_index(drop=True)
    logs['iki'] = logs.groupby('id')['down_time'].diff()

    rows = []
    for essay_id, essay in logs.groupby('id'):
        essay    = essay.reset_index(drop=True)
        n        = len(essay)
        gaps     = essay.loc[essay['activity'] == 'Input', 'iki'].dropna()
        all_gaps = essay['iki'].dropna()

        # ── VOLUME ─────────────────────────────────────────────────────────────
        final_word_count = essay['word_count'].max()
        total_events     = n

        # ── SPEED & FLUENCY ────────────────────────────────────────────────────
        median_iki_ms = gaps.median() if len(gaps) > 0 else np.nan
        iki_std       = gaps.std()    if len(gaps) > 0 else np.nan

        total_time_ms  = essay['down_time'].max() - essay['down_time'].min()
        active_time_ms = max(total_time_ms - all_gaps[all_gaps > 2_000].sum(), 1)
        typing_wpm     = final_word_count / (active_time_ms / 60_000)

        # ── PAUSING ────────────────────────────────────────────────────────────
        pauses_over_2s     = (all_gaps > 2_000).sum()
        pauses_over_5s     = (all_gaps > 5_000).sum()
        long_pauses        = all_gaps[all_gaps > 2_000]
        mean_long_pause_ms = long_pauses.mean() if len(long_pauses) > 0 else 0.0

        # ── REVISION ───────────────────────────────────────────────────────────
        frontier              = essay['cursor_position'].cummax()
        dist_from_frontier    = frontier - essay['cursor_position']
        deletion_ratio        = (essay['activity'] == 'Remove/Cut').sum() / n
        global_revision_count = (dist_from_frontier > 50).sum()
        local_revision_count  = (dist_from_frontier <= 10).sum()

        # ── PUNCTUATION ────────────────────────────────────────────────────────
        tc              = essay['text_change']
        period_count    = (tc == '.').sum()
        comma_count     = (tc == ',').sum()
        semicolon_count = (tc == ';').sum()
        newline_count   = (tc == '\n').sum()

        # ── WRITING MOMENTUM ──────────────────────────────────────────────────
        t_min   = essay['down_time'].min()
        t_max   = essay['down_time'].max()
        t_range = max(t_max - t_min, 1)
        t1, t2  = t_min + t_range / 3, t_min + 2 * t_range / 3

        wpm_early        = wpm_in_window(essay, t_min, t1)
        wpm_middle       = wpm_in_window(essay, t1,   t2)
        wpm_late         = wpm_in_window(essay, t2,   t_max)
        wpm_acceleration = wpm_middle - wpm_early

        rows.append({
            'id':                      essay_id,
            'final_word_count':        final_word_count,
            'total_events':            total_events,
            'median_iki_ms':           median_iki_ms,
            'iki_std':                 iki_std,
            'typing_wpm':             typing_wpm,
            'pauses_over_2s':          pauses_over_2s,
            'pauses_over_5s':          pauses_over_5s,
            'mean_long_pause_ms':      mean_long_pause_ms,
            'deletion_ratio':          deletion_ratio,
            'global_revision_count':   global_revision_count,
            'local_revision_count':    local_revision_count,
            'period_count':            period_count,
            'comma_count':             comma_count,
            'semicolon_count':         semicolon_count,
            'newline_count':          newline_count,
            'wpm_early':              wpm_early,
            'wpm_middle':             wpm_middle,
            'wpm_late':               wpm_late,
            'wpm_acceleration':       wpm_acceleration,
        })

    return pd.DataFrame(rows)
