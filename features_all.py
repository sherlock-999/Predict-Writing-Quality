"""
Feature Engineering: Linking Writing Process to Writing Quality
================================================================
Centralized feature computation module.

Features are derived from keystroke log analysis and reconstructed essay text.
They are organised into 8 categories:

    1. Count by values     - exact occurrence counts per activity / key / character
    2. Input word stats    - word-length distribution from typed q-sequences
    3. Numerical aggs      - full stat suite on raw numeric columns
    4. Categorical uniques - diversity of event types used
    5. Idle time           - inter-key latency bands (release → press gap)
    6. P-bursts            - production burst lengths (Input + Remove/Cut runs < 2 s apart)
    7. R-bursts            - revision burst lengths (consecutive Remove/Cut runs)
    8. Essay structure     - word / sentence / paragraph stats from reconstructed text
    9. Efficiency          - product-to-keys ratio and keys-per-second

To add a new feature:
    1. Compute it inside compute_features() under the right section
    2. Add the name to FEATURE_COLS
    3. Add it to CATEGORIES with the matching category string
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import skew as _skew

# ── Constants used during feature extraction ────────────────────────────────

ACTIVITIES = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']

EVENTS = [
    'q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick',
    'ArrowLeft', '.', ',', 'ArrowDown', 'ArrowUp', 'Enter',
    'CapsLock', "'", 'Delete', 'Unidentified',
]

TEXT_CHANGES = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']

NUM_COLS = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']

_AGG_FUNCS = [
    'count', 'mean', 'min', 'max', 'first', 'last',
    lambda x: x.quantile(0.25),
    'median',
    lambda x: x.quantile(0.75),
    'sum',
]
_AGG_NAMES = ['count', 'mean', 'min', 'max', 'first', 'last', 'q1', 'median', 'q3', 'sum']

# ── Feature column list ──────────────────────────────────────────────────────

FEATURE_COLS = [
    # 1. Count by values
    'activity_Input_cnt',
    'activity_RemoveCut_cnt',
    'activity_Nonproduction_cnt',
    'activity_Replace_cnt',
    'activity_Paste_cnt',
    'text_change_q_cnt',
    'text_change_space_cnt',
    'text_change_period_cnt',
    'text_change_comma_cnt',
    'text_change_newline_cnt',
    'text_change_quote_cnt',
    'text_change_dquote_cnt',
    'text_change_dash_cnt',
    'text_change_question_cnt',
    'text_change_semicolon_cnt',
    'down_event_q_cnt',
    'down_event_Space_cnt',
    'down_event_Backspace_cnt',
    'down_event_Shift_cnt',
    'down_event_ArrowRight_cnt',
    'down_event_Leftclick_cnt',
    'down_event_ArrowLeft_cnt',
    'down_event_period_cnt',
    'down_event_comma_cnt',
    'down_event_ArrowDown_cnt',
    'down_event_ArrowUp_cnt',
    'down_event_Enter_cnt',
    'down_event_CapsLock_cnt',
    'down_event_quote_cnt',
    'down_event_Delete_cnt',
    'down_event_Unidentified_cnt',
    'up_event_q_cnt',
    'up_event_Space_cnt',
    'up_event_Backspace_cnt',
    'up_event_Shift_cnt',
    'up_event_ArrowRight_cnt',
    'up_event_Leftclick_cnt',
    'up_event_ArrowLeft_cnt',
    'up_event_period_cnt',
    'up_event_comma_cnt',
    'up_event_ArrowDown_cnt',
    'up_event_ArrowUp_cnt',
    'up_event_Enter_cnt',
    'up_event_CapsLock_cnt',
    'up_event_quote_cnt',
    'up_event_Delete_cnt',
    'up_event_Unidentified_cnt',

    # 2. Input word stats
    'input_word_count',
    'input_word_length_mean',
    'input_word_length_max',
    'input_word_length_std',
    'input_word_length_median',
    'input_word_length_skew',

    # 3. Numerical column aggregations
    'down_time_mean', 'down_time_std', 'down_time_min', 'down_time_max',
    'down_time_median', 'down_time_q1', 'down_time_q3',
    'up_time_mean', 'up_time_std', 'up_time_min', 'up_time_max',
    'up_time_median', 'up_time_q1', 'up_time_q3',
    'action_time_sum', 'action_time_mean', 'action_time_std',
    'action_time_min', 'action_time_max', 'action_time_median',
    'action_time_q1', 'action_time_q3',
    'cursor_position_mean', 'cursor_position_std', 'cursor_position_min',
    'cursor_position_max', 'cursor_position_median', 'cursor_position_q1', 'cursor_position_q3',
    'word_count_mean', 'word_count_std', 'word_count_min',
    'word_count_max', 'word_count_median', 'word_count_q1', 'word_count_q3',

    # 4. Categorical unique counts
    'n_unique_activity',
    'n_unique_down_event',
    'n_unique_up_event',
    'n_unique_text_change',

    # 5. Idle time (release → press gap)
    'idle_largest_latency',
    'idle_median_latency',
    'idle_mean',
    'idle_std',
    'idle_total',
    'pauses_half_sec',
    'pauses_1_sec',
    'pauses_1_half_sec',
    'pauses_2_sec',
    'pauses_3_sec',

    # 6. P-bursts (production bursts)
    'p_burst_count',
    'p_burst_mean',
    'p_burst_std',
    'p_burst_median',
    'p_burst_max',
    'p_burst_first',
    'p_burst_last',

    # 7. R-bursts (revision bursts)
    'r_burst_mean',
    'r_burst_std',
    'r_burst_median',
    'r_burst_max',
    'r_burst_first',
    'r_burst_last',

    # 8. Essay structure — word features
    'word_len_count', 'word_len_mean', 'word_len_min', 'word_len_max',
    'word_len_first', 'word_len_last', 'word_len_q1', 'word_len_median',
    'word_len_q3', 'word_len_sum',

    # 8. Essay structure — sentence features
    'sent_count',
    'sent_len_mean', 'sent_len_min', 'sent_len_max',
    'sent_len_first', 'sent_len_last', 'sent_len_q1', 'sent_len_median',
    'sent_len_q3', 'sent_len_sum',
    'sent_word_count_mean', 'sent_word_count_min', 'sent_word_count_max',
    'sent_word_count_first', 'sent_word_count_last', 'sent_word_count_q1',
    'sent_word_count_median', 'sent_word_count_q3', 'sent_word_count_sum',

    # 8. Essay structure — paragraph features
    'paragraph_count',
    'paragraph_len_mean', 'paragraph_len_min', 'paragraph_len_max',
    'paragraph_len_first', 'paragraph_len_last', 'paragraph_len_q1',
    'paragraph_len_median', 'paragraph_len_q3', 'paragraph_len_sum',
    'paragraph_word_count_mean', 'paragraph_word_count_min', 'paragraph_word_count_max',
    'paragraph_word_count_first', 'paragraph_word_count_last',
    'paragraph_word_count_q1', 'paragraph_word_count_median',
    'paragraph_word_count_q3', 'paragraph_word_count_sum',

    # 9. Efficiency
    'product_to_keys',
    'keys_per_second',
]

# ── Category mapping ─────────────────────────────────────────────────────────

CATEGORIES = {
    # 1. Count by values
    'activity_Input_cnt':           'Count by Values',
    'activity_RemoveCut_cnt':       'Count by Values',
    'activity_Nonproduction_cnt':   'Count by Values',
    'activity_Replace_cnt':         'Count by Values',
    'activity_Paste_cnt':           'Count by Values',
    'text_change_q_cnt':            'Count by Values',
    'text_change_space_cnt':        'Count by Values',
    'text_change_period_cnt':       'Count by Values',
    'text_change_comma_cnt':        'Count by Values',
    'text_change_newline_cnt':      'Count by Values',
    'text_change_quote_cnt':        'Count by Values',
    'text_change_dquote_cnt':       'Count by Values',
    'text_change_dash_cnt':         'Count by Values',
    'text_change_question_cnt':     'Count by Values',
    'text_change_semicolon_cnt':    'Count by Values',
    'down_event_q_cnt':             'Count by Values',
    'down_event_Space_cnt':         'Count by Values',
    'down_event_Backspace_cnt':     'Count by Values',
    'down_event_Shift_cnt':         'Count by Values',
    'down_event_ArrowRight_cnt':    'Count by Values',
    'down_event_Leftclick_cnt':     'Count by Values',
    'down_event_ArrowLeft_cnt':     'Count by Values',
    'down_event_period_cnt':        'Count by Values',
    'down_event_comma_cnt':         'Count by Values',
    'down_event_ArrowDown_cnt':     'Count by Values',
    'down_event_ArrowUp_cnt':       'Count by Values',
    'down_event_Enter_cnt':         'Count by Values',
    'down_event_CapsLock_cnt':      'Count by Values',
    'down_event_quote_cnt':         'Count by Values',
    'down_event_Delete_cnt':        'Count by Values',
    'down_event_Unidentified_cnt':  'Count by Values',
    'up_event_q_cnt':               'Count by Values',
    'up_event_Space_cnt':           'Count by Values',
    'up_event_Backspace_cnt':       'Count by Values',
    'up_event_Shift_cnt':           'Count by Values',
    'up_event_ArrowRight_cnt':      'Count by Values',
    'up_event_Leftclick_cnt':       'Count by Values',
    'up_event_ArrowLeft_cnt':       'Count by Values',
    'up_event_period_cnt':          'Count by Values',
    'up_event_comma_cnt':           'Count by Values',
    'up_event_ArrowDown_cnt':       'Count by Values',
    'up_event_ArrowUp_cnt':         'Count by Values',
    'up_event_Enter_cnt':           'Count by Values',
    'up_event_CapsLock_cnt':        'Count by Values',
    'up_event_quote_cnt':           'Count by Values',
    'up_event_Delete_cnt':          'Count by Values',
    'up_event_Unidentified_cnt':    'Count by Values',

    # 2. Input word stats
    'input_word_count':             'Input Word Stats',
    'input_word_length_mean':       'Input Word Stats',
    'input_word_length_max':        'Input Word Stats',
    'input_word_length_std':        'Input Word Stats',
    'input_word_length_median':     'Input Word Stats',
    'input_word_length_skew':       'Input Word Stats',

    # 3. Numerical aggs
    'down_time_mean':               'Numerical Aggs',
    'down_time_std':                'Numerical Aggs',
    'down_time_min':                'Numerical Aggs',
    'down_time_max':                'Numerical Aggs',
    'down_time_median':             'Numerical Aggs',
    'down_time_q1':                 'Numerical Aggs',
    'down_time_q3':                 'Numerical Aggs',
    'up_time_mean':                 'Numerical Aggs',
    'up_time_std':                  'Numerical Aggs',
    'up_time_min':                  'Numerical Aggs',
    'up_time_max':                  'Numerical Aggs',
    'up_time_median':               'Numerical Aggs',
    'up_time_q1':                   'Numerical Aggs',
    'up_time_q3':                   'Numerical Aggs',
    'action_time_sum':              'Numerical Aggs',
    'action_time_mean':             'Numerical Aggs',
    'action_time_std':              'Numerical Aggs',
    'action_time_min':              'Numerical Aggs',
    'action_time_max':              'Numerical Aggs',
    'action_time_median':           'Numerical Aggs',
    'action_time_q1':               'Numerical Aggs',
    'action_time_q3':               'Numerical Aggs',
    'cursor_position_mean':         'Numerical Aggs',
    'cursor_position_std':          'Numerical Aggs',
    'cursor_position_min':          'Numerical Aggs',
    'cursor_position_max':          'Numerical Aggs',
    'cursor_position_median':       'Numerical Aggs',
    'cursor_position_q1':           'Numerical Aggs',
    'cursor_position_q3':           'Numerical Aggs',
    'word_count_mean':              'Numerical Aggs',
    'word_count_std':               'Numerical Aggs',
    'word_count_min':               'Numerical Aggs',
    'word_count_max':               'Numerical Aggs',
    'word_count_median':            'Numerical Aggs',
    'word_count_q1':                'Numerical Aggs',
    'word_count_q3':                'Numerical Aggs',

    # 4. Categorical uniques
    'n_unique_activity':            'Categorical Uniques',
    'n_unique_down_event':          'Categorical Uniques',
    'n_unique_up_event':            'Categorical Uniques',
    'n_unique_text_change':         'Categorical Uniques',

    # 5. Idle time
    'idle_largest_latency':         'Idle Time',
    'idle_median_latency':          'Idle Time',
    'idle_mean':                    'Idle Time',
    'idle_std':                     'Idle Time',
    'idle_total':                   'Idle Time',
    'pauses_half_sec':              'Idle Time',
    'pauses_1_sec':                 'Idle Time',
    'pauses_1_half_sec':            'Idle Time',
    'pauses_2_sec':                 'Idle Time',
    'pauses_3_sec':                 'Idle Time',

    # 6. P-bursts
    'p_burst_count':                'P-Bursts',
    'p_burst_mean':                 'P-Bursts',
    'p_burst_std':                  'P-Bursts',
    'p_burst_median':               'P-Bursts',
    'p_burst_max':                  'P-Bursts',
    'p_burst_first':                'P-Bursts',
    'p_burst_last':                 'P-Bursts',

    # 7. R-bursts
    'r_burst_mean':                 'R-Bursts',
    'r_burst_std':                  'R-Bursts',
    'r_burst_median':               'R-Bursts',
    'r_burst_max':                  'R-Bursts',
    'r_burst_first':                'R-Bursts',
    'r_burst_last':                 'R-Bursts',

    # 8. Essay structure
    'word_len_count':               'Essay Structure',
    'word_len_mean':                'Essay Structure',
    'word_len_min':                 'Essay Structure',
    'word_len_max':                 'Essay Structure',
    'word_len_first':               'Essay Structure',
    'word_len_last':                'Essay Structure',
    'word_len_q1':                  'Essay Structure',
    'word_len_median':              'Essay Structure',
    'word_len_q3':                  'Essay Structure',
    'word_len_sum':                 'Essay Structure',
    'sent_count':                   'Essay Structure',
    'sent_len_mean':                'Essay Structure',
    'sent_len_min':                 'Essay Structure',
    'sent_len_max':                 'Essay Structure',
    'sent_len_first':               'Essay Structure',
    'sent_len_last':                'Essay Structure',
    'sent_len_q1':                  'Essay Structure',
    'sent_len_median':              'Essay Structure',
    'sent_len_q3':                  'Essay Structure',
    'sent_len_sum':                 'Essay Structure',
    'sent_word_count_mean':         'Essay Structure',
    'sent_word_count_min':          'Essay Structure',
    'sent_word_count_max':          'Essay Structure',
    'sent_word_count_first':        'Essay Structure',
    'sent_word_count_last':         'Essay Structure',
    'sent_word_count_q1':           'Essay Structure',
    'sent_word_count_median':       'Essay Structure',
    'sent_word_count_q3':           'Essay Structure',
    'sent_word_count_sum':          'Essay Structure',
    'paragraph_count':              'Essay Structure',
    'paragraph_len_mean':           'Essay Structure',
    'paragraph_len_min':            'Essay Structure',
    'paragraph_len_max':            'Essay Structure',
    'paragraph_len_first':          'Essay Structure',
    'paragraph_len_last':           'Essay Structure',
    'paragraph_len_q1':             'Essay Structure',
    'paragraph_len_median':         'Essay Structure',
    'paragraph_len_q3':             'Essay Structure',
    'paragraph_len_sum':            'Essay Structure',
    'paragraph_word_count_mean':    'Essay Structure',
    'paragraph_word_count_min':     'Essay Structure',
    'paragraph_word_count_max':     'Essay Structure',
    'paragraph_word_count_first':   'Essay Structure',
    'paragraph_word_count_last':    'Essay Structure',
    'paragraph_word_count_q1':      'Essay Structure',
    'paragraph_word_count_median':  'Essay Structure',
    'paragraph_word_count_q3':      'Essay Structure',
    'paragraph_word_count_sum':     'Essay Structure',

    # 9. Efficiency
    'product_to_keys':              'Efficiency',
    'keys_per_second':              'Efficiency',
}

CAT_PALETTE = {
    'Count by Values':    '#4C72B0',
    'Input Word Stats':   '#DD8452',
    'Numerical Aggs':     '#55A868',
    'Categorical Uniques':'#C44E52',
    'Idle Time':          '#8172B2',
    'P-Bursts':           '#937860',
    'R-Bursts':           '#DA8BC3',
    'Essay Structure':    '#CCB974',
    'Efficiency':         '#64B5CD',
}

# ── Private helpers ──────────────────────────────────────────────────────────

def _safe(vals, func, default=0.0):
    """Apply func to a list; return default if the list is empty."""
    return func(vals) if len(vals) > 0 else default


def _count_val(series, value):
    """Count exact matches of value in a pandas Series."""
    return int((series == value).sum())


def _reconstruct_essay(group):
    """
    Replay keystroke events to recover the final essay text.

    Parameters
    ----------
    group : pd.DataFrame
        Rows for one essay (Nonproduction events already filtered out),
        columns: activity, cursor_position, text_change.
    """
    text = ""
    for _, row in group[['activity', 'cursor_position', 'text_change']].iterrows():
        act, cur, tc = row['activity'], row['cursor_position'], row['text_change']

        if act == 'Replace':
            parts = tc.split(' => ')
            if len(parts) == 2:
                old, new = parts
                text = text[:cur - len(new)] + new + text[cur - len(new) + len(old):]

        elif act == 'Paste':
            text = text[:cur - len(tc)] + tc + text[cur - len(tc):]

        elif act == 'Remove/Cut':
            text = text[:cur] + text[cur + len(tc):]

        elif act.startswith('Move From'):
            try:
                crp = act[10:]
                spl = crp.split(' To ')
                vals = [v.split(', ') for v in spl]
                x1, y1 = int(vals[0][0][1:]), int(vals[0][1][:-1])
                x2, y2 = int(vals[1][0][1:]), int(vals[1][1][:-1])
                if x1 != x2:
                    if x1 < x2:
                        text = (text[:x1] + text[y1:y2] +
                                text[x1:y1] + text[y2:])
                    else:
                        text = (text[:x2] + text[x1:y1] +
                                text[x2:x1] + text[y1:])
            except Exception:
                pass

        else:
            text = text[:cur - len(tc)] + tc + text[cur - len(tc):]

    return text


def _agg_series(values, prefix):
    """
    Compute [count, mean, min, max, first, last, q1, median, q3, sum]
    for a list of numeric values and return a dict with prefixed keys.
    """
    if len(values) == 0:
        return {f'{prefix}_{s}': 0.0 for s in _AGG_NAMES}
    arr = np.array(values, dtype=float)
    return {
        f'{prefix}_count':  len(arr),
        f'{prefix}_mean':   arr.mean(),
        f'{prefix}_min':    arr.min(),
        f'{prefix}_max':    arr.max(),
        f'{prefix}_first':  arr[0],
        f'{prefix}_last':   arr[-1],
        f'{prefix}_q1':     np.percentile(arr, 25),
        f'{prefix}_median': np.median(arr),
        f'{prefix}_q3':     np.percentile(arr, 75),
        f'{prefix}_sum':    arr.sum(),
    }


def _burst_sizes(mask_series):
    """
    Given a boolean Series, return a list of lengths of consecutive True runs.
    """
    sizes = []
    run   = 0
    for v in mask_series:
        if v:
            run += 1
        else:
            if run > 0:
                sizes.append(run)
                run = 0
    if run > 0:
        sizes.append(run)
    return sizes


# ── Main computation ─────────────────────────────────────────────────────────

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

        # ── 1. COUNT BY VALUES ───────────────────────────────────────────────
        _act = essay['activity']
        _tc  = essay['text_change']
        _de  = essay['down_event']
        _ue  = essay['up_event']

        act_map = [
            ('Input',         'activity_Input_cnt'),
            ('Remove/Cut',    'activity_RemoveCut_cnt'),
            ('Nonproduction', 'activity_Nonproduction_cnt'),
            ('Replace',       'activity_Replace_cnt'),
            ('Paste',         'activity_Paste_cnt'),
        ]
        for val, key in act_map:
            row[key] = _count_val(_act, val)

        tc_map = [
            ('q',   'text_change_q_cnt'),
            (' ',   'text_change_space_cnt'),
            ('.',   'text_change_period_cnt'),
            (',',   'text_change_comma_cnt'),
            ('\n',  'text_change_newline_cnt'),
            ("'",   'text_change_quote_cnt'),
            ('"',   'text_change_dquote_cnt'),
            ('-',   'text_change_dash_cnt'),
            ('?',   'text_change_question_cnt'),
            (';',   'text_change_semicolon_cnt'),
        ]
        for val, key in tc_map:
            row[key] = _count_val(_tc, val)

        de_map = [
            ('q',            'down_event_q_cnt'),
            ('Space',        'down_event_Space_cnt'),
            ('Backspace',    'down_event_Backspace_cnt'),
            ('Shift',        'down_event_Shift_cnt'),
            ('ArrowRight',   'down_event_ArrowRight_cnt'),
            ('Leftclick',    'down_event_Leftclick_cnt'),
            ('ArrowLeft',    'down_event_ArrowLeft_cnt'),
            ('.',            'down_event_period_cnt'),
            (',',            'down_event_comma_cnt'),
            ('ArrowDown',    'down_event_ArrowDown_cnt'),
            ('ArrowUp',      'down_event_ArrowUp_cnt'),
            ('Enter',        'down_event_Enter_cnt'),
            ('CapsLock',     'down_event_CapsLock_cnt'),
            ("'",            'down_event_quote_cnt'),
            ('Delete',       'down_event_Delete_cnt'),
            ('Unidentified', 'down_event_Unidentified_cnt'),
        ]
        for val, key in de_map:
            row[key] = _count_val(_de, val)

        ue_map = [
            ('q',            'up_event_q_cnt'),
            ('Space',        'up_event_Space_cnt'),
            ('Backspace',    'up_event_Backspace_cnt'),
            ('Shift',        'up_event_Shift_cnt'),
            ('ArrowRight',   'up_event_ArrowRight_cnt'),
            ('Leftclick',    'up_event_Leftclick_cnt'),
            ('ArrowLeft',    'up_event_ArrowLeft_cnt'),
            ('.',            'up_event_period_cnt'),
            (',',            'up_event_comma_cnt'),
            ('ArrowDown',    'up_event_ArrowDown_cnt'),
            ('ArrowUp',      'up_event_ArrowUp_cnt'),
            ('Enter',        'up_event_Enter_cnt'),
            ('CapsLock',     'up_event_CapsLock_cnt'),
            ("'",            'up_event_quote_cnt'),
            ('Delete',       'up_event_Delete_cnt'),
            ('Unidentified', 'up_event_Unidentified_cnt'),
        ]
        for val, key in ue_map:
            row[key] = _count_val(_ue, val)

        # ── 2. INPUT WORD STATS ──────────────────────────────────────────────
        # Concatenate all typed characters (excluding replace arrows and NoChange),
        # then extract runs of 'q' — each run represents one typed word.
        inp = essay[
            ~essay['text_change'].str.contains('=>', na=False) &
            (essay['text_change'] != 'NoChange')
        ]['text_change'].str.cat(sep='')
        word_seqs = re.findall(r'q+', inp)
        wlens     = [len(w) for w in word_seqs]

        row['input_word_count']          = len(wlens)
        row['input_word_length_mean']    = _safe(wlens, np.mean)
        row['input_word_length_max']     = _safe(wlens, np.max)
        row['input_word_length_std']     = _safe(wlens, np.std)
        row['input_word_length_median']  = _safe(wlens, np.median)
        row['input_word_length_skew']    = _safe(wlens, _skew)

        # ── 3. NUMERICAL COLUMN AGGREGATIONS ────────────────────────────────
        # action_time: always include sum (total key-hold time)
        at = essay['action_time']
        row['action_time_sum']    = float(at.sum())
        row['action_time_mean']   = float(at.mean())
        row['action_time_std']    = float(at.std())
        row['action_time_min']    = float(at.min())
        row['action_time_max']    = float(at.max())
        row['action_time_median'] = float(at.median())
        row['action_time_q1']     = float(at.quantile(0.25))
        row['action_time_q3']     = float(at.quantile(0.75))

        for col in ['down_time', 'up_time', 'cursor_position', 'word_count']:
            s = essay[col]
            row[f'{col}_mean']   = float(s.mean())
            row[f'{col}_std']    = float(s.std())
            row[f'{col}_min']    = float(s.min())
            row[f'{col}_max']    = float(s.max())
            row[f'{col}_median'] = float(s.median())
            row[f'{col}_q1']     = float(s.quantile(0.25))
            row[f'{col}_q3']     = float(s.quantile(0.75))

        # ── 4. CATEGORICAL UNIQUE COUNTS ────────────────────────────────────
        row['n_unique_activity']    = int(essay['activity'].nunique())
        row['n_unique_down_event']  = int(essay['down_event'].nunique())
        row['n_unique_up_event']    = int(essay['up_event'].nunique())
        row['n_unique_text_change'] = int(essay['text_change'].nunique())

        # ── 5. IDLE TIME (release → press gap) ──────────────────────────────
        # gap = next key's down_time − previous key's up_time (in seconds)
        # Restricted to Input and Remove/Cut events only.
        idle_df = essay.copy()
        idle_df['up_time_lagged'] = idle_df['up_time'].shift(1)
        idle_df['time_diff'] = (
            (idle_df['down_time'] - idle_df['up_time_lagged']).abs() / 1000
        ).fillna(0)
        idle_df = idle_df[idle_df['activity'].isin(['Input', 'Remove/Cut'])]
        td = idle_df['time_diff'].dropna()

        row['idle_largest_latency'] = float(td.max()) if len(td) else 0.0
        row['idle_median_latency']  = float(td.median()) if len(td) else 0.0
        row['idle_mean']            = float(td.mean()) if len(td) else 0.0
        row['idle_std']             = float(td.std())  if len(td) else 0.0
        row['idle_total']           = float(td.sum())  if len(td) else 0.0
        row['pauses_half_sec']      = int(((td > 0.5) & (td < 1.0)).sum())
        row['pauses_1_sec']         = int(((td > 1.0) & (td < 1.5)).sum())
        row['pauses_1_half_sec']    = int(((td > 1.5) & (td < 2.0)).sum())
        row['pauses_2_sec']         = int(((td > 2.0) & (td < 3.0)).sum())
        row['pauses_3_sec']         = int((td > 3.0).sum())

        # ── 6. P-BURSTS (production bursts) ─────────────────────────────────
        # A P-burst = consecutive Input or Remove/Cut events where the idle
        # gap between each consecutive pair is < 2 seconds.
        pb_df = essay.copy()
        pb_df['up_time_lagged'] = pb_df['up_time'].shift(1)
        pb_df['time_diff'] = (
            (pb_df['down_time'] - pb_df['up_time_lagged']).abs() / 1000
        ).fillna(0)
        pb_df = pb_df[pb_df['activity'].isin(['Input', 'Remove/Cut'])].copy()
        pb_df['in_burst'] = pb_df['time_diff'] < 2
        pb_sizes = _burst_sizes(pb_df['in_burst'])

        row['p_burst_count']  = len(pb_sizes)
        row['p_burst_mean']   = _safe(pb_sizes, np.mean)
        row['p_burst_std']    = _safe(pb_sizes, np.std)
        row['p_burst_median'] = _safe(pb_sizes, np.median)
        row['p_burst_max']    = _safe(pb_sizes, np.max)
        row['p_burst_first']  = pb_sizes[0]  if pb_sizes else 0
        row['p_burst_last']   = pb_sizes[-1] if pb_sizes else 0

        # ── 7. R-BURSTS (revision bursts) ───────────────────────────────────
        # An R-burst = consecutive Remove/Cut events within the pool of
        # Input + Remove/Cut events.
        rb_df = essay[essay['activity'].isin(['Input', 'Remove/Cut'])].copy()
        rb_df['is_remove'] = rb_df['activity'] == 'Remove/Cut'
        rb_sizes = _burst_sizes(rb_df['is_remove'])

        row['r_burst_mean']   = _safe(rb_sizes, np.mean)
        row['r_burst_std']    = _safe(rb_sizes, np.std)
        row['r_burst_median'] = _safe(rb_sizes, np.median)
        row['r_burst_max']    = _safe(rb_sizes, np.max)
        row['r_burst_first']  = rb_sizes[0]  if rb_sizes else 0
        row['r_burst_last']   = rb_sizes[-1] if rb_sizes else 0

        # ── 8. ESSAY STRUCTURE ───────────────────────────────────────────────
        # Reconstruct the essay text by replaying all non-Nonproduction events.
        non_np = essay[essay['activity'] != 'Nonproduction']
        essay_text = _reconstruct_essay(non_np)

        # Word features — split on spaces, newlines, sentence-end punctuation
        raw_words = [w for w in re.split(r'[ \n\.\?\!]', essay_text) if w]
        wl_vals   = [len(w) for w in raw_words]
        row.update(_agg_series(wl_vals, 'word_len'))

        # Sentence features — split on [.?!]
        raw_sents      = [s.replace('\n', '').strip()
                          for s in re.split(r'[\.\?\!]', essay_text)
                          if s.strip()]
        sent_lens      = [len(s) for s in raw_sents]
        sent_wc        = [len(s.split()) for s in raw_sents]
        row['sent_count'] = len(raw_sents)
        row.update(_agg_series(sent_lens, 'sent_len'))
        # remove the redundant sent_len_count (already stored as sent_count)
        row.pop('sent_len_count', None)
        row.update(_agg_series(sent_wc, 'sent_word_count'))
        row.pop('sent_word_count_count', None)

        # Paragraph features — split on newline
        raw_paras  = [p.strip() for p in essay_text.split('\n') if p.strip()]
        para_lens  = [len(p) for p in raw_paras]
        para_wc    = [len(p.split()) for p in raw_paras]
        row['paragraph_count'] = len(raw_paras)
        row.update(_agg_series(para_lens, 'paragraph_len'))
        row.pop('paragraph_len_count', None)
        row.update(_agg_series(para_wc, 'paragraph_word_count'))
        row.pop('paragraph_word_count_count', None)

        # ── 9. EFFICIENCY ────────────────────────────────────────────────────
        keys_pressed     = int(essay['activity'].isin(['Input', 'Remove/Cut']).sum())
        session_seconds  = (essay['up_time'].max() - essay['down_time'].min()) / 1000
        product_len      = len(essay_text)

        row['product_to_keys']  = product_len / max(keys_pressed, 1)
        row['keys_per_second']  = keys_pressed / max(session_seconds, 1)

        rows.append(row)

    result = pd.DataFrame(rows)

    # Ensure every declared feature column is present (fill missing with 0)
    for col in FEATURE_COLS:
        if col not in result.columns:
            result[col] = 0.0

    return result[['id'] + FEATURE_COLS]