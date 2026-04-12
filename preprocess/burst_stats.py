"""
Burst Statistics
================
Features based on consecutive runs (bursts) of related events.

Two burst types:
  P-burst (production burst)
      A consecutive run of Input or Remove/Cut events where each gap between
      the release of one key and the press of the next is < 2 seconds.
      Captures uninterrupted writing flow episodes.

  R-burst (revision burst)
      A consecutive run of Remove/Cut events within the pool of
      Input + Remove/Cut events (i.e. ignoring Nonproduction events).
      Captures focused deletion episodes.

Processing order inside compute_features():
    Step 4 : idle_features(essay)      — idle gaps and pause counts
    Step 5 : p_burst_features(essay)   — production burst lengths
    Step 6 : r_burst_features(essay)   — revision burst lengths
"""

import numpy as np
import pandas as pd


# ── Private helper ────────────────────────────────────────────────────────────

def _burst_sizes(bool_series: pd.Series) -> list:
    """Return a list of lengths of consecutive True runs in a boolean Series."""
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


# ── Public feature functions ──────────────────────────────────────────────────

def idle_features(essay: pd.DataFrame) -> dict:
    """
    Inter-key idle gap features (time from key release to next key press).

    Computed only on Input and Remove/Cut events; navigation events are excluded
    because their gaps reflect deliberate pauses, not cognitive load from writing.
    Gaps are in seconds.

    Pause bands are mutually exclusive ranges, not cumulative counts:
      pauses_half_sec    : 0.5 s < gap < 1.0 s
      pauses_1_sec       : 1.0 s < gap < 1.5 s
      pauses_1_half_sec  : 1.5 s < gap < 2.0 s
      pauses_2_sec       : 2.0 s < gap < 3.0 s
      pauses_3_sec       : gap > 3.0 s

    Returns
    -------
    dict
        idle_{largest_latency, median_latency, mean, std, total}
        pauses_{half_sec, 1_sec, 1_half_sec, 2_sec, 3_sec}
    """
    df = essay.copy()
    df['up_time_lagged'] = df['up_time'].shift(1)
    df['gap_sec'] = ((df['down_time'] - df['up_time_lagged']).abs() / 1000).fillna(0)
    df = df[df['activity'].isin(['Input', 'Remove/Cut'])]
    gaps = df['gap_sec'].dropna()

    return {
        'idle_largest_latency': float(gaps.max())    if len(gaps) else 0.0,
        'idle_median_latency':  float(gaps.median()) if len(gaps) else 0.0,
        'idle_mean':            float(gaps.mean())   if len(gaps) else 0.0,
        'idle_std':             float(gaps.std())    if len(gaps) else 0.0,
        'idle_total':           float(gaps.sum())    if len(gaps) else 0.0,
        'pauses_half_sec':      int(((gaps > 0.5) & (gaps < 1.0)).sum()),
        'pauses_1_sec':         int(((gaps > 1.0) & (gaps < 1.5)).sum()),
        'pauses_1_half_sec':    int(((gaps > 1.5) & (gaps < 2.0)).sum()),
        'pauses_2_sec':         int(((gaps > 2.0) & (gaps < 3.0)).sum()),
        'pauses_3_sec':         int((gaps > 3.0).sum()),
    }


def p_burst_features(essay: pd.DataFrame) -> dict:
    """
    Production burst features.

    A P-burst is a run of consecutive Input or Remove/Cut events where the
    idle gap between each pair is < 2 seconds. Longer bursts indicate
    sustained, uninterrupted writing flow.

    Returns
    -------
    dict
        p_burst_{count, mean, std, median, max, first, last}
    """
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
    """
    Revision burst features.

    An R-burst is a run of consecutive Remove/Cut events within the pool of
    Input + Remove/Cut events. A large R-burst means the writer deleted a lot
    at once — likely a structural revision rather than a typo fix.

    Returns
    -------
    dict
        r_burst_{mean, std, median, max, first}   (r_burst_last dropped, gain < 15)
    """
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
