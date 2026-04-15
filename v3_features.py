"""
Testing New Features: Linking Writing Process to Writing Quality
================================================================
Extends testing_features.py with:

  Table 1 additions:
    verbosity             — total keystrokes (Input + Remove/Cut combined)
    idle_smallest_latency — minimum inter-key idle gap
    initial_pause         — time before first keystroke (down_time_min as proxy)
    verbosity_per_word    — verbosity ÷ word count

  Table 2 additions — Time-Sensitive Keystroke Indices (9. new category):
    Session divided into WINDOW_SEC-second windows; events = Input+Remove/Cut per window.
    ts_events_std      — StDev of events across windows
    ts_slope           — slope of linear regression on the event time series
    ts_entropy         — Shannon entropy of the normalised event distribution
    ts_uniformity      — 1 − JSD(actual ∥ uniform); 1 = perfectly uniform typing rate
    ts_local_extremes  — number of direction changes in the event time series
    ts_recurrence_mean — mean gap between windows that contain at least one event
    ts_recurrence_std  — std of those gaps

Pipeline stages (sequential)
-----------------------------
  Step 1  count_features          — count specific keys, activities, text changes
  Step 2  input_word_features     — word-length stats from typed q-sequences
  Step 3  timing_features         — action/down/up time, cursor, word-count aggs
  Step 4  idle_features           — inter-key idle gaps, pause counts, smallest latency
  Step 5  p_burst_features        — production burst lengths (writing flow episodes)
  Step 6  r_burst_features        — revision burst lengths (deletion episodes)
  Step 7  reconstruct_essay       — replay keystrokes to recover final essay text
  Step 8  word_features           — word length stats from reconstructed text
  Step 9  sentence_features       — sentence length and word-count stats
  Step 10 paragraph_features      — paragraph length and word-count stats
  Step 11 efficiency_features     — product-to-keys ratio and keys-per-second
  Step 12 verbosity + initial_pause
  Step 13 session_duration_sec
  Step 14 normalize_counts        — per-word normalisation of count-based features
  Step 15 time_window_features    — time-series statistics over fixed-width windows

Usage:
    from testing_new_features import compute_features, FEATURE_COLS, CATEGORIES, CAT_PALETTE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import numpy as np
import pandas as pd
from scipy.stats import skew as _skew

from preprocess.essay_reconstruction import reconstruct_essay
from preprocess.keystroke_stats      import count_features, input_word_features, timing_features
from preprocess.text_stats           import word_features, sentence_features, paragraph_features
from preprocess.efficiency           import efficiency_features


# =============================================================================
# BURST / IDLE FEATURES  (extended — adds idle_smallest_latency)
# =============================================================================

def _burst_sizes(bool_series: pd.Series) -> list:
    sizes, run = [], 0
    for v in bool_series:
        if v:
            run += 1
        elif run:
            sizes.append(run)
            run = 0
    if run:
        sizes.append(run)
    return sizes


def idle_features_extended(essay: pd.DataFrame) -> dict:
    """
    Inter-key idle gap features — extends burst_stats.idle_features with:
      idle_smallest_latency : minimum gap (Table 1: Smallest Latency)

    Gap = (next down_time − prev up_time) / 1000, restricted to Input + Remove/Cut.
    """
    df = essay.copy()
    df['up_time_lagged'] = df['up_time'].shift(1)
    df['gap_sec'] = ((df['down_time'] - df['up_time_lagged']).abs() / 1000).fillna(0)
    df = df[df['activity'].isin(['Input', 'Remove/Cut'])]
    gaps = df['gap_sec'].dropna()

    return {
        'idle_largest_latency':  float(gaps.max())    if len(gaps) else 0.0,
        'idle_smallest_latency': float(gaps.min())    if len(gaps) else 0.0,  # NEW
        'idle_median_latency':   float(gaps.median()) if len(gaps) else 0.0,
        'idle_mean':             float(gaps.mean())   if len(gaps) else 0.0,
        'idle_std':              float(gaps.std())    if len(gaps) else 0.0,
        'idle_total':            float(gaps.sum())    if len(gaps) else 0.0,
        'pauses_half_sec':       int(((gaps > 0.5) & (gaps < 1.0)).sum()),
        'pauses_1_sec':          int(((gaps > 1.0) & (gaps < 1.5)).sum()),
        'pauses_1_half_sec':     int(((gaps > 1.5) & (gaps < 2.0)).sum()),
        'pauses_2_sec':          int(((gaps > 2.0) & (gaps < 3.0)).sum()),
        'pauses_3_sec':          int((gaps > 3.0).sum()),
    }


def p_burst_features(essay: pd.DataFrame) -> dict:
    df = essay.copy()
    df['up_time_lagged'] = df['up_time'].shift(1)
    df['gap_sec'] = ((df['down_time'] - df['up_time_lagged']).abs() / 1000).fillna(0)
    df = df[df['activity'].isin(['Input', 'Remove/Cut'])].copy()
    df['in_burst'] = df['gap_sec'] < 2

    sizes = _burst_sizes(df['in_burst'])

    return {
        'p_burst_count':  len(sizes),
        'p_burst_mean':   float(np.mean(sizes))   if sizes else 0.0,
        'p_burst_std':    float(np.std(sizes))    if sizes else 0.0,
        'p_burst_median': float(np.median(sizes)) if sizes else 0.0,
        'p_burst_max':    float(np.max(sizes))    if sizes else 0.0,
        'p_burst_first':  float(sizes[0])         if sizes else 0.0,
        'p_burst_last':   float(sizes[-1])        if sizes else 0.0,
    }


def r_burst_features(essay: pd.DataFrame) -> dict:
    df = essay[essay['activity'].isin(['Input', 'Remove/Cut'])].copy()
    df['is_remove'] = df['activity'] == 'Remove/Cut'

    sizes = _burst_sizes(df['is_remove'])

    return {
        'r_burst_mean':   float(np.mean(sizes))   if sizes else 0.0,
        'r_burst_std':    float(np.std(sizes))    if sizes else 0.0,
        'r_burst_median': float(np.median(sizes)) if sizes else 0.0,
        'r_burst_max':    float(np.max(sizes))    if sizes else 0.0,
        'r_burst_first':  float(sizes[0])         if sizes else 0.0,
    }


# =============================================================================
# NORMALISATION  (per-word — removes essay-length bias from count features)
# =============================================================================

def _normalize_counts(row: dict) -> dict:
    word_count = max(row.get('word_len_count', 1), 1)

    return {
        # Verbosity
        'verbosity_per_word':    row.get('verbosity', 0)                  / word_count,  # NEW
        # Punctuation & Formatting
        'comma_per_word':        row.get('down_event_comma_cnt', 0)       / word_count,
        'comma_text_per_word':   row.get('text_change_comma_cnt', 0)      / word_count,
        'period_per_word':       row.get('down_event_period_cnt', 0)      / word_count,
        'period_text_per_word':  row.get('text_change_period_cnt', 0)     / word_count,
        'enter_per_word':        row.get('down_event_Enter_cnt', 0)       / word_count,
        'newline_per_word':      row.get('text_change_newline_cnt', 0)    / word_count,
        'shift_per_word':        row.get('down_event_Shift_cnt', 0)       / word_count,
        'capslock_per_word':     row.get('down_event_CapsLock_cnt', 0)    / word_count,
        'dash_per_word':         row.get('text_change_dash_cnt', 0)       / word_count,
        'question_per_word':     row.get('text_change_question_cnt', 0)   / word_count,
        'quote_per_word':        row.get('text_change_quote_cnt', 0)      / word_count,
        # Typing Speed
        'q_keys_per_word':       row.get('down_event_q_cnt', 0)           / word_count,
        'q_text_per_word':       row.get('text_change_q_cnt', 0)          / word_count,
        'space_per_word':        row.get('down_event_Space_cnt', 0)       / word_count,
        'space_text_per_word':   row.get('text_change_space_cnt', 0)      / word_count,
        # Pausing Behavior
        'pauses_half_per_word':  row.get('pauses_half_sec', 0)            / word_count,
        'pauses_1_per_word':     row.get('pauses_1_sec', 0)               / word_count,
        'pauses_1half_per_word': row.get('pauses_1_half_sec', 0)          / word_count,
        'pauses_2_per_word':     row.get('pauses_2_sec', 0)               / word_count,
        'pauses_3_per_word':     row.get('pauses_3_sec', 0)               / word_count,
        # Revision Behavior
        'removecut_per_word':    row.get('activity_RemoveCut_cnt', 0)     / word_count,
        'backspace_per_word':    row.get('down_event_Backspace_cnt', 0)   / word_count,
        # Production Flow
        'input_per_word':        row.get('activity_Input_cnt', 0)         / word_count,
        'nonproduction_per_word':row.get('activity_Nonproduction_cnt', 0) / word_count,
        'leftclick_per_word':    row.get('down_event_Leftclick_cnt', 0)   / word_count,
        'arrowleft_per_word':    row.get('down_event_ArrowLeft_cnt', 0)   / word_count,
        'arrowright_per_word':   row.get('down_event_ArrowRight_cnt', 0)  / word_count,
        # Sentence & Burst
        'sent_per_word':         row.get('sent_count', 0)                 / word_count,
        'p_burst_per_word':      row.get('p_burst_count', 0)              / word_count,
    }


# =============================================================================
# TIME-SENSITIVE KEYSTROKE INDICES  (Table 2)
# =============================================================================

WINDOW_SEC = 60  # width of each time window in seconds


def time_window_features(essay: pd.DataFrame, window_sec: int = WINDOW_SEC) -> dict:
    """
    Divide the session into fixed-width time windows and compute statistics
    over the resulting event-count time series.

    Only Input and Remove/Cut events are counted per window (productive keystrokes).

    Parameters
    ----------
    essay      : all rows for one essay, sorted by event_id
    window_sec : width of each window in seconds (default 60)

    Returns
    -------
    dict
        ts_events_std, ts_slope, ts_entropy, ts_uniformity,
        ts_local_extremes, ts_recurrence_mean, ts_recurrence_std
    """
    # Active keystrokes only; timestamps in seconds relative to session start
    active = essay[essay['activity'].isin(['Input', 'Remove/Cut'])].copy()
    if active.empty:
        return {k: 0.0 for k in [
            'ts_events_std', 'ts_slope', 'ts_entropy', 'ts_uniformity',
            'ts_local_extremes', 'ts_recurrence_mean', 'ts_recurrence_std',
        ]}

    t0       = essay['down_time'].min()
    t_sec    = (active['down_time'] - t0) / 1000
    duration = (essay['down_time'].max() - t0) / 1000

    n_windows  = max(int(np.ceil(duration / window_sec)), 1)
    window_idx = (t_sec // window_sec).astype(int).clip(0, n_windows - 1)
    counts     = np.zeros(n_windows, dtype=float)
    for idx in window_idx:
        counts[idx] += 1

    # ── StDev Events ─────────────────────────────────────────────────────────
    ts_events_std = float(np.std(counts))

    # ── Slope ─────────────────────────────────────────────────────────────────
    # Slope of OLS line fitted to (window_index, event_count).
    # Positive = accelerating writing; negative = decelerating.
    xs         = np.arange(n_windows, dtype=float)
    ts_slope   = float(np.polyfit(xs, counts, 1)[0]) if n_windows > 1 else 0.0

    # ── Shannon Entropy ───────────────────────────────────────────────────────
    # H = -Σ p_i * log(p_i), p_i = counts_i / total_events
    # 0 if all events in one window; max = log(n_windows) if perfectly spread.
    total = counts.sum()
    if total > 0 and n_windows > 1:
        p          = counts / total
        p_nz       = p[p > 0]
        ts_entropy = float(-np.sum(p_nz * np.log(p_nz)))
    else:
        ts_entropy = 0.0

    # ── Degree of Uniformity (Jensen-Shannon Divergence) ─────────────────────
    # JSD between actual distribution P and uniform Q = [1/n, …, 1/n].
    # Bounded [0, log(2)]; we normalise to [0, 1] then return uniformity = 1 − JSD_norm.
    # uniformity = 1 → perfectly uniform typing; 0 → all events in one window.
    if total > 0 and n_windows > 1:
        p   = counts / total
        q   = np.ones(n_windows) / n_windows
        m   = (p + q) / 2
        def _kl(a, b):
            mask = a > 0
            return np.sum(a[mask] * np.log(a[mask] / b[mask]))
        jsd          = (_kl(p, m) + _kl(q, m)) / 2
        jsd_norm     = jsd / np.log(2)          # normalise to [0, 1]
        ts_uniformity = float(1.0 - jsd_norm)
    else:
        ts_uniformity = 0.0

    # ── Local Extremes ────────────────────────────────────────────────────────
    # Count direction reversals in the event-count time series.
    # A reversal is where consecutive differences change sign (↑ then ↓ or vice versa).
    if n_windows > 2:
        diff            = np.diff(counts)
        sign_changes    = int(np.sum(diff[:-1] * diff[1:] < 0))
        ts_local_extremes = sign_changes
    else:
        ts_local_extremes = 0

    # ── Average & StdDev Recurrence ───────────────────────────────────────────
    # Recurrence = gap (in windows) between consecutive active windows − 1.
    # 0 if every window has ≥1 event; increases when pauses skip whole windows.
    active_windows = np.where(counts > 0)[0]
    if len(active_windows) > 1:
        gaps                = np.diff(active_windows) - 1   # 0 = adjacent windows
        ts_recurrence_mean  = float(gaps.mean())
        ts_recurrence_std   = float(gaps.std())
    else:
        ts_recurrence_mean  = 0.0
        ts_recurrence_std   = 0.0

    return {
        'ts_events_std':      ts_events_std,
        'ts_slope':           ts_slope,
        'ts_entropy':         ts_entropy,
        'ts_uniformity':      ts_uniformity,
        'ts_local_extremes':  float(ts_local_extremes),
        'ts_recurrence_mean': ts_recurrence_mean,
        'ts_recurrence_std':  ts_recurrence_std,
    }


# =============================================================================
# COMPUTE FEATURES
# =============================================================================

def compute_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-essay features from raw keystroke logs.

    Parameters
    ----------
    logs : pd.DataFrame
        Raw logs with columns: id, event_id, down_time, up_time, action_time,
        activity, down_event, up_event, text_change, cursor_position, word_count.

    Returns
    -------
    pd.DataFrame
        One row per essay; columns = ['id'] + FEATURE_COLS.
    """
    logs = logs.sort_values(['id', 'event_id']).reset_index(drop=True)

    rows = []
    for essay_id, essay in logs.groupby('id'):
        essay = essay.reset_index(drop=True)
        row   = {'id': essay_id}

        # Step 1 — count specific keys, activities, and text-change characters
        row.update(count_features(essay))

        # Step 2 — word-length stats from typed q-sequences (no reconstruction needed)
        row.update(input_word_features(essay))

        # Step 3 — aggregate action_time, down_time, up_time, cursor, word_count
        row.update(timing_features(essay))

        # Step 4 — idle gaps (extended: includes idle_smallest_latency)
        row.update(idle_features_extended(essay))

        # Step 5 — production burst lengths (runs of typing < 2 s apart)
        row.update(p_burst_features(essay))

        # Step 6 — revision burst lengths (consecutive Remove/Cut runs)
        row.update(r_burst_features(essay))

        # Step 7 — replay all events to recover the final essay text
        non_nonproduction = essay[essay['activity'] != 'Nonproduction']
        text = reconstruct_essay(non_nonproduction)

        # Step 8 — word length stats from the reconstructed essay text
        row.update(word_features(text))

        # Step 9 — sentence length and word-count stats from reconstructed text
        row.update(sentence_features(text))

        # Step 10 — paragraph length and word-count stats from reconstructed text
        row.update(paragraph_features(text))

        # Step 11 — efficiency ratios (need both essay df and reconstructed text)
        row.update(efficiency_features(essay, text))

        # Step 12 — verbosity (Table 1: total keystrokes = Input + Remove/Cut)
        #            initial_pause (Table 1: down_time_min = time before first keystroke)
        row['verbosity']     = row.get('activity_Input_cnt', 0) + row.get('activity_RemoveCut_cnt', 0)
        row['initial_pause'] = row.get('down_time_min', 0) / 1000  # ms → seconds

        # Step 13 — session duration in seconds
        row['session_duration_sec'] = (row.get('down_time_max', 0) - row.get('down_time_min', 0)) / 1000

        # Step 14 — normalise count-based features per word
        row.update(_normalize_counts(row))

        # Step 15 — time-series statistics over fixed-width windows
        row.update(time_window_features(essay))

        rows.append(row)

    result = pd.DataFrame(rows).fillna(0)
    return result[['id'] + FEATURE_COLS]


# =============================================================================
# FEATURE COLUMNS
# =============================================================================

FEATURE_COLS = [

    # ── 1. WORD CHARACTERISTICS ───────────────────────────────────────────────
    'word_len_sum',               # total chars across all words — strongest essay-length proxy
    'word_len_mean',              # mean word length — indicator of vocabulary richness
    'word_len_q3',                # 75th pct word length — how long the longer words are
    'word_len_max',               # longest single word used
    'word_len_first',             # length of the first word typed
    'word_len_last',              # length of the last word typed
    'word_len_count',             # number of words — essentially the final word count
    'input_word_count',           # distinct words typed during the session
    'input_word_length_mean',     # mean typed word length (from keystroke sequence)
    'input_word_length_max',      # longest word typed during the session
    'input_word_length_std',      # std of typed word lengths — vocabulary variety
    'input_word_length_skew',     # skew of typed word lengths — long-tail word usage
    'word_count_std',             # std of running word count — pace of word accumulation
    'word_count_median',          # median running word count over the session
    'word_count_q1',              # 25th pct word count — how much was written early on
    'word_count_q3',              # 75th pct word count — word count near end of session

    # ── 2. SENTENCE CHARACTERISTICS ───────────────────────────────────────────
    'sent_len_mean',              # mean sentence length — verbose vs. concise style
    'sent_len_median',            # median sentence length — robust to outlier sentences
    'sent_len_min',               # shortest sentence — detects fragments
    'sent_len_max',               # longest sentence — captures complex compound structures
    'sent_len_first',             # length of the opening sentence
    'sent_len_last',              # length of the closing sentence
    'sent_len_q1',                # 25th pct sentence length
    'sent_len_q3',                # 75th pct sentence length
    'sent_per_word',              # number of sentences per word
    'sent_word_count_mean',       # mean words per sentence — syntactic density
    'sent_word_count_max',        # longest sentence by word count
    'sent_word_count_first',      # words in the first sentence
    'sent_word_count_last',       # words in the last sentence
    'sent_word_count_q1',         # 25th pct words per sentence
    'sent_word_count_median',     # median words per sentence
    'sent_word_count_q3',         # 75th pct words per sentence

    # ── 3. PARAGRAPH CHARACTERISTICS ──────────────────────────────────────────
    'paragraph_len_mean',         # mean paragraph length — how developed each idea is
    'paragraph_len_median',       # median paragraph length — robust central tendency
    'paragraph_len_min',          # shortest paragraph — thin or transitional paragraphs
    'paragraph_len_max',          # longest paragraph — most developed argument
    'paragraph_len_first',        # length of the introduction paragraph
    'paragraph_len_last',         # length of the conclusion paragraph
    'paragraph_len_q1',           # 25th pct paragraph length
    'paragraph_len_q3',           # 75th pct paragraph length
    'paragraph_count',            # total number of paragraphs — structural organisation
    'paragraph_word_count_mean',  # mean words per paragraph
    'paragraph_word_count_min',   # fewest words in any paragraph
    'paragraph_word_count_max',   # most words in any paragraph
    'paragraph_word_count_first', # word count of the introduction
    'paragraph_word_count_last',  # word count of the conclusion
    'paragraph_word_count_q1',    # 25th pct paragraph word count
    'paragraph_word_count_median',# median paragraph word count
    'paragraph_word_count_q3',    # 75th pct paragraph word count

    # ── 4. PUNCTUATION & FORMATTING ───────────────────────────────────────────
    'comma_per_word',             # comma key presses per word
    'comma_text_per_word',        # literal ',' inserted per word
    'period_per_word',            # period key presses per word
    'period_text_per_word',       # literal '.' inserted per word
    'enter_per_word',             # enter presses per word
    'newline_per_word',           # '\n' inserted per word
    'shift_per_word',             # shift presses per word
    'capslock_per_word',          # caps-lock usage per word
    'dash_per_word',              # '-' inserted per word
    'question_per_word',          # '?' inserted per word
    'quote_per_word',             # "'" inserted per word

    # ── 5. TYPING SPEED & FLUENCY ─────────────────────────────────────────────
    'verbosity',                  # total keystrokes (Input + Remove/Cut) — Table 1
    'verbosity_per_word',         # total keystrokes per word — length-normalised verbosity
    'keys_per_second',            # gross keystrokes per second — overall typing speed
    'action_time_sum',            # total time keys were physically held down
    'action_time_mean',           # mean key-hold duration — typical keystroke rhythm
    'action_time_std',            # std of hold durations — consistency of key presses
    'action_time_median',         # median hold duration — robust rhythm measure
    'action_time_max',            # longest single key hold (e.g. held backspace to delete)
    'action_time_q1',             # 25th pct hold duration
    'action_time_q3',             # 75th pct hold duration
    'q_keys_per_word',            # letter keystrokes per word
    'q_text_per_word',            # anonymised letter insertions per word
    'space_per_word',             # space presses per word
    'space_text_per_word',        # spaces inserted per word
    'n_unique_text_change',       # distinct text changes — typing content variety
    'n_unique_down_event',        # distinct keys used — keyboard range

    # ── 6. PAUSING BEHAVIOR ───────────────────────────────────────────────────
    'initial_pause',              # time before first keystroke in seconds — Table 1
    'idle_largest_latency',       # single longest idle gap — Table 1: Largest Latency
    'idle_smallest_latency',      # single shortest idle gap — Table 1: Smallest Latency (NEW)
    'idle_median_latency',        # median idle gap — Table 1: Median Latency
    'idle_mean',                  # mean idle gap — overall session pacing
    'idle_std',                   # std of idle gaps — consistency of typing rhythm
    'idle_total',                 # total time spent idle — time not actively typing
    'pauses_half_per_word',       # pauses > 0.5 s per word — Table 1: 0.5 Second Pauses
    'pauses_1_per_word',          # pauses > 1 s per word — Table 1: 1 Second Pauses
    'pauses_1half_per_word',      # pauses > 1.5 s per word — Table 1: 1.5 Second Pauses
    'pauses_2_per_word',          # pauses > 2 s per word — Table 1: 2 Second Pauses
    'pauses_3_per_word',          # pauses > 3 s per word — Table 1: 3 Second Pauses

    # ── 7. REVISION BEHAVIOR ──────────────────────────────────────────────────
    'removecut_per_word',         # Remove/Cut events per word
    'backspace_per_word',         # backspace presses per word — Table 1: Backspaces
    'r_burst_mean',               # mean revision burst length — typical deletion run size
    'r_burst_std',                # std of revision burst lengths — consistency of deletions
    'r_burst_median',             # median revision burst length
    'r_burst_max',                # largest single deletion burst — big structural revision
    'r_burst_first',              # first revision burst — how soon writer started correcting

    # ── 8. PRODUCTION FLOW & NAVIGATION ──────────────────────────────────────
    'input_per_word',             # input events per word
    'nonproduction_per_word',     # navigation/thinking events per word
    'p_burst_per_word',           # production bursts per word
    'p_burst_mean',               # mean burst length — typical flow episode size
    'p_burst_std',                # std of burst lengths — consistency of flow
    'p_burst_median',             # median burst length
    'p_burst_max',                # longest single burst — peak flow episode
    'p_burst_first',              # first burst length — how fast writer entered flow
    'p_burst_last',               # last burst length — late-session production effort
    'product_to_keys',            # net chars produced ÷ total keys pressed — efficiency
    'session_duration_sec',       # total session duration in seconds
    'cursor_position_mean',       # mean cursor position — average depth into essay
    'cursor_position_std',        # std of cursor position — how much cursor moved around
    'cursor_position_median',     # median cursor position
    'cursor_position_q1',         # 25th pct cursor position
    'cursor_position_q3',         # 75th pct cursor position
    'leftclick_per_word',         # mouse clicks per word
    'arrowleft_per_word',         # left-arrow presses per word
    'arrowright_per_word',        # right-arrow presses per word
    'down_time_min',              # timestamp of first keypress — session start offset
    'down_time_mean',             # mean keypress timestamp — session pacing centroid
    'down_time_std',              # std of keypress timestamps — spread of activity
    'down_time_median',           # median keypress timestamp
    'down_time_q1',               # 25th pct keypress time
    'down_time_q3',               # 75th pct keypress time
    'down_time_max',              # timestamp of last keypress — session end
    'up_time_min',                # first key-release time
    'up_time_max',                # last key-release time

    # ── 9. TIME-SENSITIVE KEYSTROKE INDICES ───────────────────────────────────
    # Session divided into 60-second windows; events = Input + Remove/Cut per window.
    'ts_events_std',              # StDev of events per window — typing rate variability
    'ts_slope',                   # slope of OLS on event time series — acceleration/deceleration
    'ts_entropy',                 # Shannon entropy of event distribution — spread of activity
    'ts_uniformity',              # 1 − JSD(actual ∥ uniform) — how constant the typing rate is
    'ts_local_extremes',          # direction changes in event series — writing rate inconsistency
    'ts_recurrence_mean',         # mean gap between active windows — average pause length
    'ts_recurrence_std',          # std of gaps between active windows — pause consistency
]

# =============================================================================
# CATEGORY MAPPING
# =============================================================================

CATEGORIES = {
    # 1. Word Characteristics
    'word_len_sum':               'Word Characteristics',
    'word_len_mean':              'Word Characteristics',
    'word_len_q3':                'Word Characteristics',
    'word_len_max':               'Word Characteristics',
    'word_len_first':             'Word Characteristics',
    'word_len_last':              'Word Characteristics',
    'word_len_count':             'Word Characteristics',
    'input_word_count':           'Word Characteristics',
    'input_word_length_mean':     'Word Characteristics',
    'input_word_length_max':      'Word Characteristics',
    'input_word_length_std':      'Word Characteristics',
    'input_word_length_skew':     'Word Characteristics',
    'word_count_std':             'Word Characteristics',
    'word_count_median':          'Word Characteristics',
    'word_count_q1':              'Word Characteristics',
    'word_count_q3':              'Word Characteristics',

    # 2. Sentence Characteristics
    'sent_len_mean':              'Sentence Characteristics',
    'sent_len_median':            'Sentence Characteristics',
    'sent_len_min':               'Sentence Characteristics',
    'sent_len_max':               'Sentence Characteristics',
    'sent_len_first':             'Sentence Characteristics',
    'sent_len_last':              'Sentence Characteristics',
    'sent_len_q1':                'Sentence Characteristics',
    'sent_len_q3':                'Sentence Characteristics',
    'sent_per_word':              'Sentence Characteristics',
    'sent_word_count_mean':       'Sentence Characteristics',
    'sent_word_count_max':        'Sentence Characteristics',
    'sent_word_count_first':      'Sentence Characteristics',
    'sent_word_count_last':       'Sentence Characteristics',
    'sent_word_count_q1':         'Sentence Characteristics',
    'sent_word_count_median':     'Sentence Characteristics',
    'sent_word_count_q3':         'Sentence Characteristics',

    # 3. Paragraph Characteristics
    'paragraph_len_mean':             'Paragraph Characteristics',
    'paragraph_len_median':           'Paragraph Characteristics',
    'paragraph_len_min':              'Paragraph Characteristics',
    'paragraph_len_max':              'Paragraph Characteristics',
    'paragraph_len_first':            'Paragraph Characteristics',
    'paragraph_len_last':             'Paragraph Characteristics',
    'paragraph_len_q1':               'Paragraph Characteristics',
    'paragraph_len_q3':               'Paragraph Characteristics',
    'paragraph_count':                'Paragraph Characteristics',
    'paragraph_word_count_mean':      'Paragraph Characteristics',
    'paragraph_word_count_min':       'Paragraph Characteristics',
    'paragraph_word_count_max':       'Paragraph Characteristics',
    'paragraph_word_count_first':     'Paragraph Characteristics',
    'paragraph_word_count_last':      'Paragraph Characteristics',
    'paragraph_word_count_q1':        'Paragraph Characteristics',
    'paragraph_word_count_median':    'Paragraph Characteristics',
    'paragraph_word_count_q3':        'Paragraph Characteristics',

    # 4. Punctuation & Formatting
    'comma_per_word':             'Punctuation & Formatting',
    'comma_text_per_word':        'Punctuation & Formatting',
    'period_per_word':            'Punctuation & Formatting',
    'period_text_per_word':       'Punctuation & Formatting',
    'enter_per_word':             'Punctuation & Formatting',
    'newline_per_word':           'Punctuation & Formatting',
    'shift_per_word':             'Punctuation & Formatting',
    'capslock_per_word':          'Punctuation & Formatting',
    'dash_per_word':              'Punctuation & Formatting',
    'question_per_word':          'Punctuation & Formatting',
    'quote_per_word':             'Punctuation & Formatting',

    # 5. Typing Speed & Fluency
    'verbosity':                  'Typing Speed & Fluency',
    'verbosity_per_word':         'Typing Speed & Fluency',
    'keys_per_second':            'Typing Speed & Fluency',
    'action_time_sum':            'Typing Speed & Fluency',
    'action_time_mean':           'Typing Speed & Fluency',
    'action_time_std':            'Typing Speed & Fluency',
    'action_time_median':         'Typing Speed & Fluency',
    'action_time_max':            'Typing Speed & Fluency',
    'action_time_q1':             'Typing Speed & Fluency',
    'action_time_q3':             'Typing Speed & Fluency',
    'q_keys_per_word':            'Typing Speed & Fluency',
    'q_text_per_word':            'Typing Speed & Fluency',
    'space_per_word':             'Typing Speed & Fluency',
    'space_text_per_word':        'Typing Speed & Fluency',
    'n_unique_text_change':       'Typing Speed & Fluency',
    'n_unique_down_event':        'Typing Speed & Fluency',

    # 6. Pausing Behavior
    'initial_pause':              'Pausing Behavior',
    'idle_largest_latency':       'Pausing Behavior',
    'idle_smallest_latency':      'Pausing Behavior',
    'idle_median_latency':        'Pausing Behavior',
    'idle_mean':                  'Pausing Behavior',
    'idle_std':                   'Pausing Behavior',
    'idle_total':                 'Pausing Behavior',
    'pauses_half_per_word':       'Pausing Behavior',
    'pauses_1_per_word':          'Pausing Behavior',
    'pauses_1half_per_word':      'Pausing Behavior',
    'pauses_2_per_word':          'Pausing Behavior',
    'pauses_3_per_word':          'Pausing Behavior',

    # 7. Revision Behavior
    'removecut_per_word':         'Revision Behavior',
    'backspace_per_word':         'Revision Behavior',
    'r_burst_mean':               'Revision Behavior',
    'r_burst_std':                'Revision Behavior',
    'r_burst_median':             'Revision Behavior',
    'r_burst_max':                'Revision Behavior',
    'r_burst_first':              'Revision Behavior',

    # 8. Production Flow & Navigation
    'input_per_word':             'Production Flow & Navigation',
    'nonproduction_per_word':     'Production Flow & Navigation',
    'p_burst_per_word':           'Production Flow & Navigation',
    'p_burst_mean':               'Production Flow & Navigation',
    'p_burst_std':                'Production Flow & Navigation',
    'p_burst_median':             'Production Flow & Navigation',
    'p_burst_max':                'Production Flow & Navigation',
    'p_burst_first':              'Production Flow & Navigation',
    'p_burst_last':               'Production Flow & Navigation',
    'product_to_keys':            'Production Flow & Navigation',
    'session_duration_sec':       'Production Flow & Navigation',
    'cursor_position_mean':       'Production Flow & Navigation',
    'cursor_position_std':        'Production Flow & Navigation',
    'cursor_position_median':     'Production Flow & Navigation',
    'cursor_position_q1':         'Production Flow & Navigation',
    'cursor_position_q3':         'Production Flow & Navigation',
    'leftclick_per_word':         'Production Flow & Navigation',
    'arrowleft_per_word':         'Production Flow & Navigation',
    'arrowright_per_word':        'Production Flow & Navigation',
    'down_time_min':              'Production Flow & Navigation',
    'down_time_mean':             'Production Flow & Navigation',
    'down_time_std':              'Production Flow & Navigation',
    'down_time_median':           'Production Flow & Navigation',
    'down_time_q1':               'Production Flow & Navigation',
    'down_time_q3':               'Production Flow & Navigation',
    'down_time_max':              'Production Flow & Navigation',
    'up_time_min':                'Production Flow & Navigation',
    'up_time_max':                'Production Flow & Navigation',

    # 9. Time-Sensitive Keystroke Indices
    'ts_events_std':              'Time-Sensitive Keystroke Indices',
    'ts_slope':                   'Time-Sensitive Keystroke Indices',
    'ts_entropy':                 'Time-Sensitive Keystroke Indices',
    'ts_uniformity':              'Time-Sensitive Keystroke Indices',
    'ts_local_extremes':          'Time-Sensitive Keystroke Indices',
    'ts_recurrence_mean':         'Time-Sensitive Keystroke Indices',
    'ts_recurrence_std':          'Time-Sensitive Keystroke Indices',
}

# =============================================================================
# COLOUR PALETTE
# =============================================================================

CAT_PALETTE = {
    'Word Characteristics':        '#4C72B0',  # blue
    'Sentence Characteristics':    '#DD8452',  # orange
    'Paragraph Characteristics':   '#55A868',  # green
    'Punctuation & Formatting':    '#8172B2',  # purple
    'Typing Speed & Fluency':      '#C44E52',  # red
    'Pausing Behavior':            '#937860',  # brown
    'Revision Behavior':           '#DA8BC3',  # pink
    'Production Flow & Navigation':  '#8C8C8C',  # grey
    'Time-Sensitive Keystroke Indices': '#2CA02C',  # green (distinct from Paragraph)
}
