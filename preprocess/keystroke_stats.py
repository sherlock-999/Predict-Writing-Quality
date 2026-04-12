"""
Keystroke Statistics
====================
Features derived directly from the raw keystroke log columns —
no essay reconstruction needed.

Three groups:
  1. count_features       — exact occurrence counts for activities, keys, text changes
  2. input_word_features  — word-length distribution from typed q-sequences
  3. timing_features      — aggregated stats on action_time, down_time, up_time,
                            cursor_position, and word_count columns

Processing order inside compute_features():
    Step 1 : count_features(essay)
    Step 2 : input_word_features(essay)
    Step 3 : timing_features(essay)
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import skew as _skew


# ── 1. Count features ─────────────────────────────────────────────────────────

def count_features(essay: pd.DataFrame) -> dict:
    """
    Count occurrences of specific activities, key events, and text changes.

    Only keys/characters with LightGBM gain >= 15 are included.

    Parameters
    ----------
    essay : pd.DataFrame
        All rows for one essay, sorted by event_id.

    Returns
    -------
    dict
        activity_*_cnt, down_event_*_cnt, up_event_*_cnt, text_change_*_cnt,
        n_unique_text_change, n_unique_down_event, n_unique_up_event
    """
    act = essay['activity']
    tc  = essay['text_change']
    de  = essay['down_event']
    ue  = essay['up_event']

    feats = {}

    # Activity type counts
    for value, key in [
        ('Input',         'activity_Input_cnt'),
        ('Remove/Cut',    'activity_RemoveCut_cnt'),
        ('Nonproduction', 'activity_Nonproduction_cnt'),
    ]:
        feats[key] = int((act == value).sum())

    # Text change character counts (punctuation is preserved, letters are 'q')
    for value, key in [
        ('q',  'text_change_q_cnt'),
        (' ',  'text_change_space_cnt'),
        ('.',  'text_change_period_cnt'),
        (',',  'text_change_comma_cnt'),
        ('\n', 'text_change_newline_cnt'),
        ("'",  'text_change_quote_cnt'),
        ('-',  'text_change_dash_cnt'),
        ('?',  'text_change_question_cnt'),
    ]:
        feats[key] = int((tc == value).sum())

    # Key down event counts
    for value, key in [
        ('q',         'down_event_q_cnt'),
        ('Space',     'down_event_Space_cnt'),
        ('Backspace', 'down_event_Backspace_cnt'),
        ('Shift',     'down_event_Shift_cnt'),
        ('ArrowRight','down_event_ArrowRight_cnt'),
        ('Leftclick', 'down_event_Leftclick_cnt'),
        ('ArrowLeft', 'down_event_ArrowLeft_cnt'),
        ('.',         'down_event_period_cnt'),
        (',',         'down_event_comma_cnt'),
        ('Enter',     'down_event_Enter_cnt'),
        ('CapsLock',  'down_event_CapsLock_cnt'),
    ]:
        feats[key] = int((de == value).sum())

    # Key up event counts (only the three with gain >= 15)
    for value, key in [
        ('q',    'up_event_q_cnt'),
        (',',    'up_event_comma_cnt'),
        ('Shift','up_event_Shift_cnt'),
    ]:
        feats[key] = int((ue == value).sum())

    # Key/event diversity
    feats['n_unique_text_change'] = int(tc.nunique())
    feats['n_unique_down_event']  = int(de.nunique())
    feats['n_unique_up_event']    = int(ue.nunique())

    return feats


# ── 2. Input word features ────────────────────────────────────────────────────

def input_word_features(essay: pd.DataFrame) -> dict:
    """
    Word-length distribution derived from typed q-sequences in text_change.

    Method: concatenate all plain-insertion text_change values (no '=>' replace
    arrows, no 'NoChange'), then find consecutive runs of 'q'. Each run is one
    typed word; its length represents that word's character count.

    Returns
    -------
    dict
        input_word_count, input_word_length_{mean, max, std, skew}
    """
    plain_input = essay[
        ~essay['text_change'].str.contains('=>', na=False) &
        (essay['text_change'] != 'NoChange')
    ]['text_change'].str.cat(sep='')

    word_seqs = re.findall(r'q+', plain_input)
    lengths   = [len(w) for w in word_seqs]

    return {
        'input_word_count':        len(lengths),
        'input_word_length_mean':  np.mean(lengths)   if lengths else 0.0,
        'input_word_length_max':   np.max(lengths)    if lengths else 0.0,
        'input_word_length_std':   np.std(lengths)    if lengths else 0.0,
        'input_word_length_skew':  float(_skew(lengths)) if lengths else 0.0,
    }


# ── 3. Timing features ────────────────────────────────────────────────────────

def timing_features(essay: pd.DataFrame) -> dict:
    """
    Aggregated statistics on numeric log columns.

    Columns covered:
      - action_time      : how long each key was physically held (ms)
      - down_time        : absolute timestamp of each key press (ms)
      - up_time          : absolute timestamp of each key release (ms)
      - cursor_position  : character position of cursor after the event
      - word_count       : running word count after the event

    Only aggregations with LightGBM gain >= 15 are included.

    Returns
    -------
    dict
        action_time_{sum,mean,std,median,max,q1,q3}
        down_time_{min,mean,std,median,q1,q3,max}
        up_time_{min,max}
        cursor_position_{mean,std,max,median,q1,q3}
        word_count_{max,std,median,q1,q3}
    """
    feats = {}

    at = essay['action_time']
    feats['action_time_sum']    = float(at.sum())
    feats['action_time_mean']   = float(at.mean())
    feats['action_time_std']    = float(at.std())
    feats['action_time_median'] = float(at.median())
    feats['action_time_max']    = float(at.max())
    feats['action_time_q1']     = float(at.quantile(0.25))
    feats['action_time_q3']     = float(at.quantile(0.75))

    dt = essay['down_time']
    feats['down_time_min']    = float(dt.min())
    feats['down_time_mean']   = float(dt.mean())
    feats['down_time_std']    = float(dt.std())
    feats['down_time_median'] = float(dt.median())
    feats['down_time_q1']     = float(dt.quantile(0.25))
    feats['down_time_q3']     = float(dt.quantile(0.75))
    feats['down_time_max']    = float(dt.max())

    ut = essay['up_time']
    feats['up_time_min'] = float(ut.min())
    feats['up_time_max'] = float(ut.max())

    cp = essay['cursor_position']
    feats['cursor_position_mean']   = float(cp.mean())
    feats['cursor_position_std']    = float(cp.std())
    feats['cursor_position_max']    = float(cp.max())
    feats['cursor_position_median'] = float(cp.median())
    feats['cursor_position_q1']     = float(cp.quantile(0.25))
    feats['cursor_position_q3']     = float(cp.quantile(0.75))

    wc = essay['word_count']
    feats['word_count_max']    = float(wc.max())
    feats['word_count_std']    = float(wc.std())
    feats['word_count_median'] = float(wc.median())
    feats['word_count_q1']     = float(wc.quantile(0.25))
    feats['word_count_q3']     = float(wc.quantile(0.75))

    return feats
